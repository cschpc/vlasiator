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
using namespace std;

// Globals needed to be able to compile
Logger logFile, diagnostic;
static dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry> mpiGrid;

int globalflags::bailingOut = 0;
bool globalflags::writeRestart = 0;
bool globalflags::balanceLoad = 0;
bool globalflags::doRefine=0;
bool globalflags::ionosphereJustSolved = false;

ObjectWrapper objectWrapper;

ObjectWrapper& getObjectWrapper() {
   return objectWrapper;
}

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
// END Globals

int main(int argn,char* args[]) {
   int myRank;
   int required=MPI_THREAD_FUNNELED;
   int provided;

   // After the MPI_T settings we can init MPI all right.
   MPI_Init_thread(&argn,&args,required,&provided);
   MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
   if (required > provided){
      exit(1);
   }

   phiprof::initialize();

   // Initialize simplified Fieldsolver grids.
   std::array<FsGridTools::FsSize_t, 3> fsGridDimensions = {15, 15, 15};
   std::array<bool,3> periodicity = {1, 1, 1};
   std::array<int,3> manualFsGridDecomposition = {0, 0, 0};

   FsGridCouplingInformation gridCoupling;
   FsGrid< std::array<Real, fsgrids::bfield::N_BFIELD>,     FS_STENCIL_WIDTH> perBGrid       (fsGridDimensions, MPI_COMM_WORLD, periodicity, gridCoupling, manualFsGridDecomposition);
//   FsGrid< std::array<Real, fsgrids::bfield::N_BFIELD>,     FS_STENCIL_WIDTH> perBDt2Grid    (fsGridDimensions, MPI_COMM_WORLD, periodicity, gridCoupling, manualFsGridDecomposition);
//   FsGrid< std::array<Real, fsgrids::efield::N_EFIELD>,     FS_STENCIL_WIDTH> EGrid          (fsGridDimensions, MPI_COMM_WORLD, periodicity, gridCoupling, manualFsGridDecomposition);
//   FsGrid< std::array<Real, fsgrids::efield::N_EFIELD>,     FS_STENCIL_WIDTH> EDt2Grid       (fsGridDimensions, MPI_COMM_WORLD, periodicity, gridCoupling, manualFsGridDecomposition);
//   FsGrid< std::array<Real, fsgrids::ehall::N_EHALL>,       FS_STENCIL_WIDTH> EHallGrid      (fsGridDimensions, MPI_COMM_WORLD, periodicity, gridCoupling, manualFsGridDecomposition);
//   FsGrid< std::array<Real, fsgrids::egradpe::N_EGRADPE>,   FS_STENCIL_WIDTH> EGradPeGrid    (fsGridDimensions, MPI_COMM_WORLD, periodicity, gridCoupling, manualFsGridDecomposition);
   FsGrid< std::array<Real, fsgrids::moments::N_MOMENTS>,   FS_STENCIL_WIDTH> momentsGrid    (fsGridDimensions, MPI_COMM_WORLD, periodicity, gridCoupling, manualFsGridDecomposition);
//   FsGrid< std::array<Real, fsgrids::moments::N_MOMENTS>,   FS_STENCIL_WIDTH> momentsDt2Grid (fsGridDimensions, MPI_COMM_WORLD, periodicity, gridCoupling, manualFsGridDecomposition);
   FsGrid< std::array<Real, fsgrids::dperb::N_DPERB>,       FS_STENCIL_WIDTH> dPerBGrid      (fsGridDimensions, MPI_COMM_WORLD, periodicity, gridCoupling, manualFsGridDecomposition);
   FsGrid< std::array<Real, fsgrids::dmoments::N_DMOMENTS>, FS_STENCIL_WIDTH> dMomentsGrid   (fsGridDimensions, MPI_COMM_WORLD, periodicity, gridCoupling, manualFsGridDecomposition);
//   FsGrid< std::array<Real, fsgrids::bgbfield::N_BGB>,      FS_STENCIL_WIDTH> BgBGrid        (fsGridDimensions, MPI_COMM_WORLD, periodicity, gridCoupling, manualFsGridDecomposition);
//   FsGrid< std::array<Real, fsgrids::volfields::N_VOL>,     FS_STENCIL_WIDTH> volGrid        (fsGridDimensions, MPI_COMM_WORLD, periodicity, gridCoupling, manualFsGridDecomposition);
   FsGrid< fsgrids::technical,                              FS_STENCIL_WIDTH> technicalGrid  (fsGridDimensions, MPI_COMM_WORLD, periodicity, gridCoupling, manualFsGridDecomposition);

   // Set DX, DY and DZ
   // TODO: This is currently just taking the values from cell 1, and assuming them to be
   // constant throughout the simulation.
//   perBGrid.DX = perBDt2Grid.DX = EGrid.DX = EDt2Grid.DX = EHallGrid.DX = EGradPeGrid.DX = momentsGrid.DX
//      = momentsDt2Grid.DX = dPerBGrid.DX = dMomentsGrid.DX = BgBGrid.DX = volGrid.DX = technicalGrid.DX
//      = 4000 / 3.0;
//   perBGrid.DY = perBDt2Grid.DY = EGrid.DY = EDt2Grid.DY = EHallGrid.DY = EGradPeGrid.DY = momentsGrid.DY
//      = momentsDt2Grid.DY = dPerBGrid.DY = dMomentsGrid.DY = BgBGrid.DY = volGrid.DY = technicalGrid.DY
//      = 4000 / 3.0;
//   perBGrid.DZ = perBDt2Grid.DZ = EGrid.DZ = EDt2Grid.DZ = EHallGrid.DZ = EGradPeGrid.DZ = momentsGrid.DZ
//      = momentsDt2Grid.DZ = dPerBGrid.DZ = dMomentsGrid.DZ = BgBGrid.DZ = volGrid.DZ = technicalGrid.DZ
//      = 4000 / 3.0;
//   // Set the physical start (lower left corner) X, Y, Z
//   perBGrid.physicalGlobalStart = perBDt2Grid.physicalGlobalStart = EGrid.physicalGlobalStart = EDt2Grid.physicalGlobalStart
//      = EHallGrid.physicalGlobalStart = EGradPeGrid.physicalGlobalStart = momentsGrid.physicalGlobalStart
//      = momentsDt2Grid.physicalGlobalStart = dPerBGrid.physicalGlobalStart = dMomentsGrid.physicalGlobalStart
//      = BgBGrid.physicalGlobalStart = volGrid.physicalGlobalStart = technicalGrid.physicalGlobalStart
//      = {-10000, -10000, -10000};

   perBGrid.DX = momentsGrid.DX = dPerBGrid.DX = dMomentsGrid.DX = technicalGrid.DX
      = 4000 / 3.0;
   perBGrid.DY = momentsGrid.DY = dPerBGrid.DY = dMomentsGrid.DY = technicalGrid.DX
      = 4000 / 3.0;
   perBGrid.DZ = momentsGrid.DZ = dPerBGrid.DZ = dMomentsGrid.DZ = technicalGrid.DX
      = 4000 / 3.0;
   // Set the physical start (lower left corner) X, Y, Z
   perBGrid.physicalGlobalStart = momentsGrid.physicalGlobalStart = dPerBGrid.physicalGlobalStart = dMomentsGrid.physicalGlobalStart = technicalGrid.physicalGlobalStart
      = {-10000, -10000, -10000};

   auto s = technicalGrid.getLocalSize();
   cout << "local size" << endl;
   cout << s[0] << endl;  // 15
   cout << s[1] << endl;  // 15
   cout << s[2] << endl;  // 15
   cout << "size is " << perBGrid.getData().size() << endl;  // 6859 = (15+2+2)**3

   cout << "data is " << endl;
   auto data = perBGrid.getData();
   for (auto v : data) {
     if (v[0] == 0.0 && v[1] == 0 && v[2] == 0) {
       continue;
     }
     printf("(%f,%f,%f) ", v[0], v[1], v[2]);
   }
   cout << endl;


  calculateDerivativesSimple(perBGrid, momentsGrid, dPerBGrid, dMomentsGrid, technicalGrid, true);
  //dPerBGrid.updateGhostCells();


   perBGrid.finalize();
//   perBDt2Grid.finalize();
//   EGrid.finalize();
//   EDt2Grid.finalize();
//   EHallGrid.finalize();
//   EGradPeGrid.finalize();
   momentsGrid.finalize();
//   momentsDt2Grid.finalize();
   dPerBGrid.finalize();
   dMomentsGrid.finalize();
//   BgBGrid.finalize();
//   volGrid.finalize();
   technicalGrid.finalize();

   MPI_Finalize();
   return 0;
}
