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
   int provided;


   // After the MPI_T settings we can init MPI all right.
   MPI_Init_thread(&argn,&args,required,&provided);
   MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
   if (required > provided){
      if(myRank==MASTER_RANK) {
         cerr << "(MAIN): MPI_Init_thread failed! Got " << provided << ", need "<<required <<endl;
      }
      exit(1);
   }

   phiprof::initialize();
   
   double initialWtime =  MPI_Wtime();
   SysBoundary& sysBoundaryContainer = getObjectWrapper().sysBoundaryContainer;

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

   //Get version and config info here
   std::string version;
   std::string config;
   //Only master needs the info
   if (myRank==MASTER_RANK){
      version=readparameters.versionInfo();
      config=readparameters.configInfo();
   }



   // Init parallel logger:

      if (logFile.open(MPI_COMM_WORLD,MASTER_RANK,"logfile.txt",P::isRestart) == false) {
         if(myRank == MASTER_RANK) cerr << "(MAIN) ERROR: Logger failed to open logfile!" << endl;
         exit(1);
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
   
   // Initialize simplified Fieldsolver grids.
   // Needs to be done here already ad the background field will be set right away, before going to initializeGrid even

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

   // Initialize grid.  After initializeGrid local cells have dist
   // functions, and B fields set. Cells have also been classified for
   // the various sys boundary conditions.  All remote cells have been
   // created. All spatial date computed this far is up to date for
   // FULL_NEIGHBORHOOD. Block lists up to date for
   // VLASOV_SOLVER_NEIGHBORHOOD (but dist function has not been communicated)
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
