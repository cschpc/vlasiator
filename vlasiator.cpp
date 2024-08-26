/*
 * This file is part of Vlasiator.
 * Copyright 2010-2016 Finnish Meteorological Institute
 *
 * For details of usage, see the COPYING file and read the "Rules of the Road"
 * at http://www.physics.helsinki.fi/vlasiator/
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>
#include <sstream>
#include <ctime>

#ifdef _OPENMP
   #include <omp.h>
#endif

#include <fsgrid.hpp>

#include "vlasovmover.h"
#include "definitions.h"
#include "mpiconversion.h"
#include "logger.h"
#include "parameters.h"
#include "readparameters.h"
#include "spatial_cell_wrapper.hpp"
#include "datareduction/datareducer.h"
#include "sysboundary/sysboundary.h"
#include "fieldtracing/fieldtracing.h"

#include "fieldsolver/fs_common.h"
#include "projects/project.h"
#include "grid.h"
#include "iowrite.h"
#include "ioread.h"

#include "object_wrapper.h"
#include "fieldsolver/gridGlue.hpp"
#include "fieldsolver/derivatives.hpp"

#ifdef CATCH_FPE
#include <fenv.h>
#include <signal.h>
/*! Function used to abort the program upon detecting a floating point exception. Which exceptions are caught is defined using the function feenableexcept.
 */
void fpehandler(int sig_num)
{
   signal(SIGFPE, fpehandler);
   printf("SIGFPE: floating point exception occured, exiting.\n");
   abort();
}
#endif

#include "phiprof.hpp"

Logger logFile, diagnostic;
static dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry> mpiGrid;

using namespace std;

int globalflags::bailingOut = 0;
bool globalflags::writeRestart = 0;
bool globalflags::balanceLoad = 0;
bool globalflags::doRefine=0;
bool globalflags::ionosphereJustSolved = false;

ObjectWrapper objectWrapper;

void addTimedBarrier(string name){
#ifdef NDEBUG
//let's not do a barrier
   return;
#endif
   phiprof::Timer btimer {name, {"Barriers", "MPI"}};
   MPI_Barrier(MPI_COMM_WORLD);
}


/*! Report spatial cell counts per refinement level as well as velocity cell counts per population into logfile
 */
void report_cell_and_block_counts(dccrg::Dccrg<spatial_cell::SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid){
   cint maxRefLevel = mpiGrid.get_maximum_refinement_level();
   const vector<CellID> localCells = getLocalCells();
   cint popCount = getObjectWrapper().particleSpecies.size();

   // popCount+1 as we store the spatial cell counts and then the populations' v_cell counts.
   // maxRefLevel+1 as e.g. there's 2 levels at maxRefLevel == 1
   std::vector<int64_t> localCounts((popCount+1)*(maxRefLevel+1), 0), globalCounts((popCount+1)*(maxRefLevel+1), 0);

   for (const auto cellid : localCells) {
      cint level = mpiGrid.get_refinement_level(cellid);
      localCounts[level]++;
      for(int pop=0; pop<popCount; pop++) {
         localCounts[maxRefLevel+1 + level*popCount + pop] += mpiGrid[cellid]->get_number_of_velocity_blocks(pop);
      }
   }

   MPI_Reduce(localCounts.data(), globalCounts.data(), (popCount+1)*(maxRefLevel+1), MPI_INT64_T, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);

   logFile << "(CELLS) tstep = " << P::tstep << " time = " << P::t << " spatial cells [ ";
   for(int level = 0; level <= maxRefLevel; level++) {
      logFile << globalCounts[level] << " ";
   }
   logFile << "] blocks ";
   for(int pop=0; pop<popCount; pop++) {
      logFile << getObjectWrapper().particleSpecies[pop].name << " [ ";
      for(int level = 0; level <= maxRefLevel; level++) {
         logFile << globalCounts[maxRefLevel+1 + level*popCount + pop] << " ";
      }
      logFile << "] ";
   }
   logFile << endl << flush;

}


void computeNewTimeStep(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
			FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid, Real &newDt, bool &isChanged) {

   phiprof::Timer computeTimestepTimer {"compute-timestep"};
   // Compute maximum time step. This cannot be done at the first step as the solvers compute the limits for each cell.

   isChanged = false;

   const vector<CellID>& cells = getLocalCells();
   /* Arrays for storing local (per process) and global max dt
      0th position stores ordinary space propagation dt
      1st position stores velocity space propagation dt
      2nd position stores field propagation dt
   */
   Real dtMaxLocal[3];
   Real dtMaxGlobal[3];

   dtMaxLocal[0] = numeric_limits<Real>::max();
   dtMaxLocal[1] = numeric_limits<Real>::max();
   dtMaxLocal[2] = numeric_limits<Real>::max();

   for (vector<CellID>::const_iterator cell_id = cells.begin(); cell_id != cells.end(); ++cell_id) {
      SpatialCell* cell = mpiGrid[*cell_id];
      const Real dx = cell->parameters[CellParams::DX];
      const Real dy = cell->parameters[CellParams::DY];
      const Real dz = cell->parameters[CellParams::DZ];

      cell->parameters[CellParams::MAXRDT] = numeric_limits<Real>::max();

      for (uint popID = 0; popID < getObjectWrapper().particleSpecies.size(); ++popID) {
         cell->set_max_r_dt(popID, numeric_limits<Real>::max());
         vmesh::VelocityBlockContainer<vmesh::LocalID>& blockContainer = cell->get_velocity_blocks(popID);
         const Real* blockParams = blockContainer.getParameters();
         const Real EPS = numeric_limits<Real>::min() * 1000;
         for (vmesh::LocalID blockLID = 0; blockLID < blockContainer.size(); ++blockLID) {
            for (unsigned int i = 0; i < WID; i += WID - 1) {
               const Real Vx =
                   blockParams[blockLID * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VXCRD] +
                   (i + HALF) * blockParams[blockLID * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVX] + EPS;
               const Real Vy =
                   blockParams[blockLID * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VYCRD] +
                   (i + HALF) * blockParams[blockLID * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVY] + EPS;
               const Real Vz =
                   blockParams[blockLID * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VZCRD] +
                   (i + HALF) * blockParams[blockLID * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVZ] + EPS;

               const Real dt_max_cell = min({dx / fabs(Vx), dy / fabs(Vy), dz / fabs(Vz)});
               cell->set_max_r_dt(popID, min(dt_max_cell, cell->get_max_r_dt(popID)));
            }
         }
         cell->parameters[CellParams::MAXRDT] = min(cell->get_max_r_dt(popID), cell->parameters[CellParams::MAXRDT]);
      }

      if (cell->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY ||
          (cell->sysBoundaryLayer == 1 && cell->sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY)) {
         // spatial fluxes computed also for boundary cells
         dtMaxLocal[0] = min(dtMaxLocal[0], cell->parameters[CellParams::MAXRDT]);
      }

      if (cell->parameters[CellParams::MAXVDT] != 0 &&
          (cell->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY ||
           (P::vlasovAccelerateMaxwellianBoundaries && cell->sysBoundaryFlag == sysboundarytype::MAXWELLIAN))) {
         // acceleration only done on non-boundary cells
         dtMaxLocal[1] = min(dtMaxLocal[1], cell->parameters[CellParams::MAXVDT]);
      }
   }

   // compute max dt for fieldsolver
   const std::array<FsGridTools::FsIndex_t, 3> gridDims(technicalGrid.getLocalSize());
   for (FsGridTools::FsIndex_t k = 0; k < gridDims[2]; k++) {
      for (FsGridTools::FsIndex_t j = 0; j < gridDims[1]; j++) {
         for (FsGridTools::FsIndex_t i = 0; i < gridDims[0]; i++) {
            fsgrids::technical* cell = technicalGrid.get(i, j, k);
            if (cell->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY ||
               (cell->sysBoundaryLayer == 1 && cell->sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY)) {
               dtMaxLocal[2] = min(dtMaxLocal[2], cell->maxFsDt);
            }
         }
      }
   }

   MPI_Allreduce(&(dtMaxLocal[0]), &(dtMaxGlobal[0]), 3, MPI_Type<Real>(), MPI_MIN, MPI_COMM_WORLD);

   // If any of the solvers are disabled there should be no limits in timespace from it
   if (!P::propagateVlasovTranslation)
      dtMaxGlobal[0] = numeric_limits<Real>::max();
   if (!P::propagateVlasovAcceleration)
      dtMaxGlobal[1] = numeric_limits<Real>::max();
   if (!P::propagateField)
      dtMaxGlobal[2] = numeric_limits<Real>::max();

   creal meanVlasovCFL = 0.5 * (P::vlasovSolverMaxCFL + P::vlasovSolverMinCFL);
   creal meanFieldsCFL = 0.5 * (P::fieldSolverMaxCFL + P::fieldSolverMinCFL);
   Real subcycleDt;

   // reduce/increase dt if it is too high for any of the three propagators or too low for all propagators
   if ((P::dt > dtMaxGlobal[0] * P::vlasovSolverMaxCFL ||
        P::dt > dtMaxGlobal[1] * P::vlasovSolverMaxCFL * P::maxSlAccelerationSubcycles ||
        P::dt > dtMaxGlobal[2] * P::fieldSolverMaxCFL * P::maxFieldSolverSubcycles) ||
       (P::dt < dtMaxGlobal[0] * P::vlasovSolverMinCFL &&
        P::dt < dtMaxGlobal[1] * P::vlasovSolverMinCFL * P::maxSlAccelerationSubcycles &&
        P::dt < dtMaxGlobal[2] * P::fieldSolverMinCFL * P::maxFieldSolverSubcycles)) {

      // new dt computed
      isChanged = true;

      // set new timestep to the lowest one of all interval-midpoints
      newDt = meanVlasovCFL * dtMaxGlobal[0];
      newDt = min(newDt, meanVlasovCFL * dtMaxGlobal[1] * P::maxSlAccelerationSubcycles);
      newDt = min(newDt, meanFieldsCFL * dtMaxGlobal[2] * P::maxFieldSolverSubcycles);

      logFile << "(TIMESTEP) New dt = " << newDt << " computed on step " << P::tstep << " at " << P::t
              << "s   Maximum possible dt (not including  vlasovsolver CFL " << P::vlasovSolverMinCFL << "-"
              << P::vlasovSolverMaxCFL << " or fieldsolver CFL " << P::fieldSolverMinCFL << "-" << P::fieldSolverMaxCFL
              << ") in {r, v, BE} was " << dtMaxGlobal[0] << " " << dtMaxGlobal[1] << " " << dtMaxGlobal[2] << " "
              << " Including subcycling { v, BE}  was " << dtMaxGlobal[1] * P::maxSlAccelerationSubcycles << " "
              << dtMaxGlobal[2] * P::maxFieldSolverSubcycles << " " << endl
              << writeVerbose;

      if (P::dynamicTimestep) {
         subcycleDt = newDt;
      } else {
         logFile << "(TIMESTEP) However, fixed timestep in config overrides dt = " << P::dt << endl << writeVerbose;
         subcycleDt = P::dt;
      }
   } else {
      subcycleDt = P::dt;
   }

   // Subcycle if field solver dt < global dt (including CFL) (new or old dt hence the hassle with subcycleDt
   if (meanFieldsCFL * dtMaxGlobal[2] < subcycleDt && P::propagateField) {
      P::fieldSolverSubcycles =
          min(convert<uint>(ceil(subcycleDt / (meanFieldsCFL * dtMaxGlobal[2]))), P::maxFieldSolverSubcycles);
   } else {
      P::fieldSolverSubcycles = 1;
   }
}

ObjectWrapper& getObjectWrapper() {
   return objectWrapper;
}

/** Get local cell IDs. This function creates a cached copy of the 
 * cell ID lists to significantly improve performance. The cell ID 
 * cache is recalculated every time the mesh partitioning changes.
 * @return Local cell IDs.*/
const std::vector<CellID>& getLocalCells() {
   return Parameters::localCells;
}

void recalculateLocalCellsCache() {
     {
        vector<CellID> dummy;
        dummy.swap(Parameters::localCells);
     }
   Parameters::localCells = mpiGrid.get_cells();
}

int main(int argn,char* args[]) {
   int myRank, doBailout=0;
   const creal DT_EPSILON=1e-12;
   typedef Parameters P;
   Real newDt;
   bool dtIsChanged {false};
   
   // Before MPI_Init we hardwire some settings, if we are in OpenMPI
   int required=MPI_THREAD_FUNNELED;
   int provided, resultlen;
   char mpiversion[MPI_MAX_LIBRARY_VERSION_STRING];
   bool overrideMCAompio = false;

   MPI_Get_library_version(mpiversion, &resultlen);
   string versionstr = string(mpiversion);
   stringstream mpiioMessage;

   if(versionstr.find("Open MPI") != std::string::npos) {
      #ifdef VLASIATOR_ALLOW_MCA_OMPIO
         mpiioMessage << "We detected OpenMPI but the compilation flag VLASIATOR_ALLOW_MCA_OMPIO was set so we do not override the default MCA io flag." << endl;
      #else
         overrideMCAompio = true;
         int index, count;
         char io_value[64];
         MPI_T_cvar_handle io_handle;
         
         MPI_T_init_thread(required, &provided);
         MPI_T_cvar_get_index("io", &index);
         MPI_T_cvar_handle_alloc(index, NULL, &io_handle, &count);
         MPI_T_cvar_write(io_handle, "^ompio");
         MPI_T_cvar_read(io_handle, io_value);
         MPI_T_cvar_handle_free(&io_handle);
         mpiioMessage << "We detected OpenMPI so we set the cvars value to disable ompio, MCA io: " << io_value << endl;
      #endif
   }
   
   // After the MPI_T settings we can init MPI all right.
   MPI_Init_thread(&argn,&args,required,&provided);
   MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
   if (required > provided){
      if(myRank==MASTER_RANK) {
         cerr << "(MAIN): MPI_Init_thread failed! Got " << provided << ", need "<<required <<endl;
      }
      exit(1);
   }
   if (myRank == MASTER_RANK) {
      const char* mpiioenv = std::getenv("OMPI_MCA_io");
      if(mpiioenv != nullptr) {
         std::string mpiioenvstr(mpiioenv);
         if(mpiioenvstr.find("^ompio") == std::string::npos) {
            cout << mpiioMessage.str();
         }
      }
   }

   phiprof::initialize();
   
   double initialWtime =  MPI_Wtime();
   SysBoundary& sysBoundaryContainer = getObjectWrapper().sysBoundaryContainer;
   
   #ifdef CATCH_FPE
   // WARNING FE_INEXACT is too sensitive to be used. See man fenv.
   //feenableexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW|FE_UNDERFLOW);
   feenableexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW);
   //feenableexcept(FE_DIVBYZERO|FE_INVALID);
   signal(SIGFPE, fpehandler);
   #endif

   phiprof::Timer mainTimer {"main"};
   phiprof::Timer initTimer {"Initialization"};
   phiprof::Timer readParamsTimer {"Read parameters"};

   //init parameter file reader
   Readparameters readparameters(argn,args);

   P::addParameters();

   getObjectWrapper().addParameters();

   readparameters.parse();

   P::getParameters();

   getObjectWrapper().addPopulationParameters();
   sysBoundaryContainer.addParameters();
   projects::Project::addParameters();

   Project* project = projects::createProject();
   getObjectWrapper().project = project;
   readparameters.parse(true, false); // 2nd parsing for specific population parameters
   readparameters.helpMessage(); // Call after last parse, exits after printing help if help requested
   getObjectWrapper().getParameters();
   sysBoundaryContainer.getParameters();
   project->getParameters();
   readParamsTimer.stop();

   //Get version and config info here
   std::string version;
   std::string config;
   //Only master needs the info
   if (myRank==MASTER_RANK){
      version=readparameters.versionInfo();
      config=readparameters.configInfo();
   }



   // Init parallel logger:

   phiprof::Timer openLoggerTimer {"open logFile & diagnostic"};
   //if restarting we will append to logfiles
   if(!P::writeFullBGB) {
      if (logFile.open(MPI_COMM_WORLD,MASTER_RANK,"logfile.txt",P::isRestart) == false) {
         if(myRank == MASTER_RANK) cerr << "(MAIN) ERROR: Logger failed to open logfile!" << endl;
         exit(1);
      }
   } else {
      // If we are out to write the full background field and derivatives, we don't want to overwrite the existing run's logfile.
      if (logFile.open(MPI_COMM_WORLD,MASTER_RANK,"logfile_fullbgbio.txt",false) == false) {
         if(myRank == MASTER_RANK) cerr << "(MAIN) ERROR: Logger failed to open logfile_fullbgbio!" << endl;
         exit(1);
      }
   }
   if (P::diagnosticInterval != 0) {
      if (diagnostic.open(MPI_COMM_WORLD,MASTER_RANK,"diagnostic.txt",P::isRestart) == false) {
         if(myRank == MASTER_RANK) cerr << "(MAIN) ERROR: Logger failed to open diagnostic file!" << endl;
         exit(1);
      }
   }
   {
      int mpiProcs;
      MPI_Comm_size(MPI_COMM_WORLD,&mpiProcs);
      logFile << "(MAIN) Starting simulation with " << mpiProcs << " MPI processes ";
      #ifdef _OPENMP
         logFile << "and " << omp_get_max_threads();
      #else
         logFile << "and 0";
      #endif
      logFile << " OpenMP threads per process" << endl << writeVerbose;      
   }
   openLoggerTimer.stop();
   
   // Init project
   phiprof::Timer initProjectimer {"Init project"};
   if (project->initialize() == false) {
      if(myRank == MASTER_RANK) cerr << "(MAIN): Project did not initialize correctly!" << endl;
      exit(1);
   }
   if (project->initialized() == false) {
      if (myRank == MASTER_RANK) {
         cerr << "(MAIN): Project base class was not initialized!" << endl;
         cerr << "\t Call Project::initialize() in your project's initialize()-function." << endl;
         exit(1);
      }
   }
   initProjectimer.stop();

   // Add VAMR refinement criterias:
   vamr_ref_criteria::addRefinementCriteria();

   // Initialize simplified Fieldsolver grids.
   // Needs to be done here already ad the background field will be set right away, before going to initializeGrid even
   phiprof::Timer initFsTimer {"Init fieldsolver grids"};

   std::array<FsGridTools::FsSize_t, 3> fsGridDimensions = {convert<FsGridTools::FsSize_t>(P::xcells_ini * pow(2,P::amrMaxSpatialRefLevel)),
							    convert<FsGridTools::FsSize_t>(P::ycells_ini * pow(2,P::amrMaxSpatialRefLevel)),
							    convert<FsGridTools::FsSize_t>(P::zcells_ini * pow(2,P::amrMaxSpatialRefLevel))};

   std::array<bool,3> periodicity{sysBoundaryContainer.isPeriodic(0),
                                  sysBoundaryContainer.isPeriodic(1),
                                  sysBoundaryContainer.isPeriodic(2)};

   FsGridCouplingInformation gridCoupling;
   FsGrid< std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH> perBGrid(fsGridDimensions, MPI_COMM_WORLD, periodicity,gridCoupling, P::manualFsGridDecomposition);
   FsGrid< std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH> perBDt2Grid(fsGridDimensions, MPI_COMM_WORLD, periodicity,gridCoupling, P::manualFsGridDecomposition);
   FsGrid< std::array<Real, fsgrids::efield::N_EFIELD>, FS_STENCIL_WIDTH> EGrid(fsGridDimensions, MPI_COMM_WORLD, periodicity,gridCoupling, P::manualFsGridDecomposition);
   FsGrid< std::array<Real, fsgrids::efield::N_EFIELD>, FS_STENCIL_WIDTH> EDt2Grid(fsGridDimensions, MPI_COMM_WORLD, periodicity,gridCoupling, P::manualFsGridDecomposition);
   FsGrid< std::array<Real, fsgrids::ehall::N_EHALL>, FS_STENCIL_WIDTH> EHallGrid(fsGridDimensions, MPI_COMM_WORLD, periodicity,gridCoupling, P::manualFsGridDecomposition);
   FsGrid< std::array<Real, fsgrids::egradpe::N_EGRADPE>, FS_STENCIL_WIDTH> EGradPeGrid(fsGridDimensions, MPI_COMM_WORLD, periodicity,gridCoupling, P::manualFsGridDecomposition);
   FsGrid< std::array<Real, fsgrids::moments::N_MOMENTS>, FS_STENCIL_WIDTH> momentsGrid(fsGridDimensions, MPI_COMM_WORLD, periodicity,gridCoupling, P::manualFsGridDecomposition);
   FsGrid< std::array<Real, fsgrids::moments::N_MOMENTS>, FS_STENCIL_WIDTH> momentsDt2Grid(fsGridDimensions, MPI_COMM_WORLD, periodicity,gridCoupling, P::manualFsGridDecomposition);
   FsGrid< std::array<Real, fsgrids::dperb::N_DPERB>, FS_STENCIL_WIDTH> dPerBGrid(fsGridDimensions, MPI_COMM_WORLD, periodicity,gridCoupling, P::manualFsGridDecomposition);
   FsGrid< std::array<Real, fsgrids::dmoments::N_DMOMENTS>, FS_STENCIL_WIDTH> dMomentsGrid(fsGridDimensions, MPI_COMM_WORLD, periodicity,gridCoupling, P::manualFsGridDecomposition);
   FsGrid< std::array<Real, fsgrids::bgbfield::N_BGB>, FS_STENCIL_WIDTH> BgBGrid(fsGridDimensions, MPI_COMM_WORLD, periodicity,gridCoupling, P::manualFsGridDecomposition);
   FsGrid< std::array<Real, fsgrids::volfields::N_VOL>, FS_STENCIL_WIDTH> volGrid(fsGridDimensions, MPI_COMM_WORLD, periodicity,gridCoupling, P::manualFsGridDecomposition);
   FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> technicalGrid(fsGridDimensions, MPI_COMM_WORLD, periodicity,gridCoupling, P::manualFsGridDecomposition);

   // Set DX, DY and DZ
   // TODO: This is currently just taking the values from cell 1, and assuming them to be
   // constant throughout the simulation.
   perBGrid.DX = perBDt2Grid.DX = EGrid.DX = EDt2Grid.DX = EHallGrid.DX = EGradPeGrid.DX = momentsGrid.DX
      = momentsDt2Grid.DX = dPerBGrid.DX = dMomentsGrid.DX = BgBGrid.DX = volGrid.DX = technicalGrid.DX
      = P::dx_ini / pow(2, P::amrMaxSpatialRefLevel);
   perBGrid.DY = perBDt2Grid.DY = EGrid.DY = EDt2Grid.DY = EHallGrid.DY = EGradPeGrid.DY = momentsGrid.DY
      = momentsDt2Grid.DY = dPerBGrid.DY = dMomentsGrid.DY = BgBGrid.DY = volGrid.DY = technicalGrid.DY
      = P::dy_ini / pow(2, P::amrMaxSpatialRefLevel);
   perBGrid.DZ = perBDt2Grid.DZ = EGrid.DZ = EDt2Grid.DZ = EHallGrid.DZ = EGradPeGrid.DZ = momentsGrid.DZ
      = momentsDt2Grid.DZ = dPerBGrid.DZ = dMomentsGrid.DZ = BgBGrid.DZ = volGrid.DZ = technicalGrid.DZ
      = P::dz_ini / pow(2, P::amrMaxSpatialRefLevel);
   // Set the physical start (lower left corner) X, Y, Z
   perBGrid.physicalGlobalStart = perBDt2Grid.physicalGlobalStart = EGrid.physicalGlobalStart = EDt2Grid.physicalGlobalStart
      = EHallGrid.physicalGlobalStart = EGradPeGrid.physicalGlobalStart = momentsGrid.physicalGlobalStart
      = momentsDt2Grid.physicalGlobalStart = dPerBGrid.physicalGlobalStart = dMomentsGrid.physicalGlobalStart
      = BgBGrid.physicalGlobalStart = volGrid.physicalGlobalStart = technicalGrid.physicalGlobalStart
      = {P::xmin, P::ymin, P::zmin};

   // Checking that spatial cells are cubic, otherwise field solver is incorrect (cf. derivatives in E, Hall term)
   constexpr Real uniformTolerance=1e-3;
   if ((abs((technicalGrid.DX - technicalGrid.DY) / technicalGrid.DX) >uniformTolerance) ||
       (abs((technicalGrid.DX - technicalGrid.DZ) / technicalGrid.DX) >uniformTolerance) ||
       (abs((technicalGrid.DY - technicalGrid.DZ) / technicalGrid.DY) >uniformTolerance)) {
      if (myRank == MASTER_RANK) {
         std::cerr << "WARNING: Your spatial cells seem not to be cubic. The simulation will now abort!" << std::endl;
      }
      //just abort sending SIGTERM to all tasks
      MPI_Abort(MPI_COMM_WORLD, -1);
   }
   initFsTimer.stop();

   // Initialize grid.  After initializeGrid local cells have dist
   // functions, and B fields set. Cells have also been classified for
   // the various sys boundary conditions.  All remote cells have been
   // created. All spatial date computed this far is up to date for
   // FULL_NEIGHBORHOOD. Block lists up to date for
   // VLASOV_SOLVER_NEIGHBORHOOD (but dist function has not been communicated)
   phiprof::Timer initGridsTimer {"Init grids"};
   initializeGrids(
      argn,
      args,
      mpiGrid,
      perBGrid,
      BgBGrid,
      momentsGrid,
      momentsDt2Grid,
      EGrid,
      EGradPeGrid,
      volGrid,
      technicalGrid,
      sysBoundaryContainer,
      *project
   );
   
   // There are projects that have non-uniform and non-zero perturbed B, e.g. Magnetosphere with dipole type 4.
   // For inflow cells (e.g. maxwellian), we cannot take a FSgrid perturbed B value from the templateCell,
   // because we need a copy of the value from initialization in both perBGrid and perBDt2Grid and it isn't
   // touched as we are in boundary cells for components that aren't solved. We do a straight full copy instead
   // of looping and detecting boundary types here.
   perBDt2Grid.copyData(perBGrid);

   const std::vector<CellID>& cells = getLocalCells();
   
   initGridsTimer.stop();
   
   // Initialize data reduction operators. This should be done elsewhere in order to initialize 
   // user-defined operators:
   phiprof::Timer initDROsTimer {"Init DROs"};
   DataReducer outputReducer, diagnosticReducer;

   if(P::writeFullBGB) {
      // We need the following variables for this, let's just erase and replace the entries in the list
      P::outputVariableList.clear();
      P::outputVariableList= {"fg_b_background", "fg_b_background_vol", "fg_derivs_b_background"};
   }

   initializeDataReducers(&outputReducer, &diagnosticReducer);
   initDROsTimer.stop();
   
   // Free up memory:
   readparameters.~Readparameters();

   if(P::writeFullBGB) {
      logFile << "Writing out full BGB components and derivatives and exiting." << endl << writeVerbose;

      // initialize the communicators so we can write out ionosphere grid metadata.
      SBC::ionosphereGrid.updateIonosphereCommunicator(mpiGrid, technicalGrid);

      P::systemWriteDistributionWriteStride.push_back(0);
      P::systemWriteName.push_back("bgb");
      P::systemWriteDistributionWriteXlineStride.push_back(0);
      P::systemWriteDistributionWriteYlineStride.push_back(0);
      P::systemWriteDistributionWriteZlineStride.push_back(0);
      P::systemWritePath.push_back("./");
      P::systemWriteFsGrid.push_back(true);

      for(uint si=0; si<P::systemWriteName.size(); si++) {
         P::systemWrites.push_back(0);
      }

      const bool writeGhosts = true;
      if( writeGrid(mpiGrid,
            perBGrid,
            EGrid,
            EHallGrid,
            EGradPeGrid,
            momentsGrid,
            dPerBGrid,  
            dMomentsGrid,
            BgBGrid,
            volGrid,
            technicalGrid,
            version,
            config,
            &outputReducer,
            P::systemWriteName.size()-1,
            P::restartStripeFactor,
            writeGhosts
         ) == false
      ) {
         cerr << "FAILED TO WRITE GRID AT " << __FILE__ << " " << __LINE__ << endl;
      }
      initTimer.stop();
      mainTimer.stop();
      
      phiprof::print(MPI_COMM_WORLD,"phiprof");
      
      if (myRank == MASTER_RANK) logFile << "(MAIN): Exiting." << endl << writeVerbose;
      logFile.close();
      if (P::diagnosticInterval != 0) diagnostic.close();
      
      perBGrid.finalize();
      perBDt2Grid.finalize();
      EGrid.finalize();
      EDt2Grid.finalize();
      EHallGrid.finalize();
      EGradPeGrid.finalize();
      momentsGrid.finalize();
      momentsDt2Grid.finalize();
      dPerBGrid.finalize();
      dMomentsGrid.finalize();
      BgBGrid.finalize();
      volGrid.finalize();
      technicalGrid.finalize();

      MPI_Finalize();
      return 0;
   }

   // Run the field solver once with zero dt. This will initialize
   // Fieldsolver dt limits, and also calculate volumetric B-fields.
   // At restart, all we need at this stage has been read from the restart, the rest will be recomputed in due time.
   if(P::isRestart == false) {
      propagateFields(
         perBGrid,
         perBDt2Grid,
         EGrid,
         EDt2Grid,
         EHallGrid,
         EGradPeGrid,
         momentsGrid,
         momentsDt2Grid,
         dPerBGrid,
         dMomentsGrid,
         BgBGrid,
         volGrid,
         technicalGrid,
         sysBoundaryContainer, 0.0, 1.0
      );
   }

   phiprof::Timer getFieldsTimer {"getFieldsFromFsGrid"};
   volGrid.updateGhostCells();
   getFieldsFromFsGrid(volGrid, BgBGrid, EGradPeGrid, technicalGrid, mpiGrid, cells);
   getFieldsTimer.stop();

   // Build communicator for ionosphere solving
   // If not a restart, perBGrid and dPerBGrid are up to date after propagateFields just above. Otherwise, we should compute them.
      calculateDerivativesSimple(
         perBGrid,
         perBDt2Grid,
         momentsGrid,
         momentsDt2Grid,
         dPerBGrid,
         dMomentsGrid,
         technicalGrid,
         RK_ORDER1, // Update and compute on non-dt2 grids.
         false // Don't communicate moments, they are not needed here.
      );
      dPerBGrid.updateGhostCells();
   
   mainTimer.stop();
   
   phiprof::print(MPI_COMM_WORLD,"phiprof");
   
   if (myRank == MASTER_RANK) logFile << "(MAIN): Exiting." << endl << writeVerbose;
   logFile.close();
   if (P::diagnosticInterval != 0) diagnostic.close();
   
   perBGrid.finalize();
   perBDt2Grid.finalize();
   EGrid.finalize();
   EDt2Grid.finalize();
   EHallGrid.finalize();
   EGradPeGrid.finalize();
   momentsGrid.finalize();
   momentsDt2Grid.finalize();
   dPerBGrid.finalize();
   dMomentsGrid.finalize();
   BgBGrid.finalize();
   volGrid.finalize();
   technicalGrid.finalize();

   if(overrideMCAompio) {
      MPI_T_finalize();
   }
   MPI_Finalize();
   return 0;
}
