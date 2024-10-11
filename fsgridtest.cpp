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
#include "fieldsolver/ldz_volume.hpp"

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
   FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> technicalGrid (fsGridDimensions, MPI_COMM_WORLD, periodicity, gridCoupling, manualFsGridDecomposition);

   // Set DX, DY and DZ
   // TODO: This is currently just taking the values from cell 1, and assuming them to be
   // constant throughout the simulation.
   technicalGrid.DX = 4000 / 3.0;
   technicalGrid.DY = 4000 / 3.0;
   technicalGrid.DZ = 4000 / 3.0;
   // Set the physical start (lower left corner) X, Y, Z
   technicalGrid.physicalGlobalStart = {-10000, -10000, -10000};

/*
   for (int i = 0; i < 27; i++){
     printf("%2d %2d\n", i, perBGrid.neighbour[i]);
   }
*/

   printf("sizes: %ld\n", sizeof(fsgrids::technical));
   printf("sizes: %d %d %d %d %d %d %d %d %d\n",
   fsgrids::bfield::N_BFIELD * 8,
   fsgrids::efield::N_EFIELD * 8,
   fsgrids::ehall::N_EHALL * 8,
   fsgrids::egradpe::N_EGRADPE * 8,
   fsgrids::moments::N_MOMENTS * 8,
   fsgrids::dperb::N_DPERB * 8,
   fsgrids::dmoments::N_DMOMENTS * 8,
   fsgrids::bgbfield::N_BGB * 8,
   fsgrids::volfields::N_VOL * 8);
   printf("sizes: %d %d %d %d %d %d %d %d %d\n",
   fsgrids::bfield::N_BFIELD,
   fsgrids::efield::N_EFIELD,
   fsgrids::ehall::N_EHALL,
   fsgrids::egradpe::N_EGRADPE,
   fsgrids::moments::N_MOMENTS,
   fsgrids::dperb::N_DPERB,
   fsgrids::dmoments::N_DMOMENTS,
   fsgrids::bgbfield::N_BGB,
   fsgrids::volfields::N_VOL);

   auto s = technicalGrid.getLocalSize();
   cout << "local size" << endl;
   cout << s[0] << endl;  // 15
   cout << s[1] << endl;  // 15
   cout << s[2] << endl;  // 15

   cout << "data is " << endl;

   /*
   for (int i = 0; i < s[0] + FS_STENCIL_WIDTH*2; i++) {
       for (int j = 0; j < s[1] + FS_STENCIL_WIDTH*2; j++) {
           for (int k = 0; k < s[2] + FS_STENCIL_WIDTH*2; k++) {
               for (int l = 0; l < fsgrids::bfield::N_BFIELD; l++) {
                 printf("%2d %2d %2d %2d: %.1f\n", i, j, k, l, data[i+j+k][l]);
               }
           }
       }
   }
   */



/*
   auto data = perBGrid.getData();
   for (auto v : data) {
     if (v[0] == 0.0 && v[1] == 0 && v[2] == 0) {
       continue;
     }
     printf("(%f,%f,%f) ", v[0], v[1], v[2]);
   }
   cout << endl;
*/


   auto perBDataObj       = technicalGrid.allocate_data<std::array<Real, fsgrids::bfield::N_BFIELD>>();
   auto perBDt2DataObj    = technicalGrid.allocate_data<std::array<Real, fsgrids::bfield::N_BFIELD>>();
   auto EDataObj          = technicalGrid.allocate_data<std::array<Real, fsgrids::efield::N_EFIELD>>();
   auto EDt2DataObj       = technicalGrid.allocate_data<std::array<Real, fsgrids::efield::N_EFIELD>>();
   auto EHallDataObj      = technicalGrid.allocate_data<std::array<Real, fsgrids::ehall::N_EHALL>>();
   auto EGradPeDataObj    = technicalGrid.allocate_data<std::array<Real, fsgrids::egradpe::N_EGRADPE>>();
   auto momentsDataObj    = technicalGrid.allocate_data<std::array<Real, fsgrids::moments::N_MOMENTS>>();
   auto momentsDt2DataObj = technicalGrid.allocate_data<std::array<Real, fsgrids::moments::N_MOMENTS>>();
   auto dPerBDataObj      = technicalGrid.allocate_data<std::array<Real, fsgrids::dperb::N_DPERB>>();
   auto dMomentsDataObj   = technicalGrid.allocate_data<std::array<Real, fsgrids::dmoments::N_DMOMENTS>>();
   auto BgBDataObj        = technicalGrid.allocate_data<std::array<Real, fsgrids::bgbfield::N_BGB>>();
   auto volDataObj        = technicalGrid.allocate_data<std::array<Real, fsgrids::volfields::N_VOL>>();

   auto perBData       = perBDataObj.get();
   auto perBDt2Data    = perBDt2DataObj.get();
   auto EData          = EDataObj.get();
   auto EDt2Data       = EDt2DataObj.get();
   auto EHallData      = EHallDataObj.get();
   auto EGradPeData    = EGradPeDataObj.get();
   auto momentsData    = momentsDataObj.get();
   auto momentsDt2Data = momentsDt2DataObj.get();
   auto dPerBData      = dPerBDataObj.get();
   auto dMomentsData   = dMomentsDataObj.get();
   auto BgBData        = BgBDataObj.get();
   auto volData        = volDataObj.get();

   printf("START calculateDerivativesSimple\n");
   calculateDerivativesSimple(perBData, momentsData, dPerBData, dMomentsData, technicalGrid, true);
   printf("DONE  calculateDerivativesSimple\n");

   printf("START updateGhostCells\n");
   technicalGrid.updateGhostCells(perBData);
   printf("DONE  updateGhostCells\n");

   printf("START calculateVolumeAveragedFields\n");
   calculateVolumeAveragedFields(perBData, EData, dPerBData, volData, technicalGrid);
   printf("DONE  calculateVolumeAveragedFields\n");

   technicalGrid.finalize();

   MPI_Finalize();
   return 0;
}
