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

#include "fs_common.h"
#include "ldz_volume.hpp"

#ifndef NDEBUG
   #define DEBUG_FSOLVER
#endif

using namespace std;

void calculateVolumeAveragedFields(
   FsGrid< std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH> & perBGrid,
   FsGrid< std::array<Real, fsgrids::efield::N_EFIELD>, FS_STENCIL_WIDTH> & EGrid,
   FsGrid< std::array<Real, fsgrids::dperb::N_DPERB>, FS_STENCIL_WIDTH> & dPerBGrid,
   FsGrid< std::array<Real, fsgrids::volfields::N_VOL>, FS_STENCIL_WIDTH> & volGrid,
   FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid
) {
   calculateVolumeAveragedFields(
      perBGrid.get(),
      EGrid.get(),
      dPerBGrid.get(),
      volGrid.get(),
      technicalGrid);
}


void calculateVolumeAveragedFields(
   std::array<Real, fsgrids::bfield::N_BFIELD> * perBData,
   std::array<Real, fsgrids::efield::N_EFIELD> * EData,
   std::array<Real, fsgrids::dperb::N_DPERB> * dPerBData,
   std::array<Real, fsgrids::volfields::N_VOL> * volData,
   FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid
) {
   phiprof::Timer timer {"Calculate volume averaged fields"};

   technicalGrid.parallel_for<FsStencilRght>([=](FsStencilRght s, cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
               std::array<Real, Rec::N_REC_COEFFICIENTS> perturbedCoefficients;
               auto volGrid0 = &volData[s.center];

               // Calculate reconstruction coefficients for this cell:
               // This handles domain edges so no need to skip DO_NOT_COMPUTE or OUTER_BOUNDARY_PADDING cells.
               reconstructionCoefficients(
                  s,
                  perBData,
                  dPerBData,
                  perturbedCoefficients,
                  2
               );

               // Calculate volume average of B:
               volGrid0->at(fsgrids::volfields::PERBXVOL) = perturbedCoefficients[Rec::a_0];
               volGrid0->at(fsgrids::volfields::PERBYVOL) = perturbedCoefficients[Rec::b_0];
               volGrid0->at(fsgrids::volfields::PERBZVOL) = perturbedCoefficients[Rec::c_0];

               // This avoids out of domain accesses below.
               if(sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE || sysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING) return;

               // Calculate volume average of E (FIXME NEEDS IMPROVEMENT):
               auto EGrid_i1j1k1 = &EData[s.center];
               if ( sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY ||
                    (sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY && sysBoundaryLayer == 1)
                  ) {
                  #ifdef DEBUG_FSOLVER
                  bool ok = true;
                  if (s.yRght == s.center) ok = false;
                  if (s.zRght == s.center) == NULL) ok = false;
                  if (s.yzTopRght < 0) ok = false;
                  if (ok == false) {
                     stringstream ss;
                     ss << "ERROR, got NULL neighbor in " << __FILE__ << ":" << __LINE__ << endl;
                     cerr << ss.str(); exit(1);
                  }
                  #endif

                  auto EGrid_i1j2k1 = &EData[s.yRght];
                  auto EGrid_i1j1k2 = &EData[s.zRght];
                  auto EGrid_i1j2k2 = &EData[s.yzTopRght];

                  CHECK_FLOAT(EGrid_i1j1k1->at(fsgrids::efield::EX));
                  CHECK_FLOAT(EGrid_i1j2k1->at(fsgrids::efield::EX));
                  CHECK_FLOAT(EGrid_i1j1k2->at(fsgrids::efield::EX));
                  CHECK_FLOAT(EGrid_i1j2k2->at(fsgrids::efield::EX));
                  volGrid0->at(fsgrids::volfields::EXVOL) = FOURTH*(EGrid_i1j1k1->at(fsgrids::efield::EX) + EGrid_i1j2k1->at(fsgrids::efield::EX) + EGrid_i1j1k2->at(fsgrids::efield::EX) + EGrid_i1j2k2->at(fsgrids::efield::EX));
                  CHECK_FLOAT(volGrid0->at(fsgrids::volfields::EXVOL));
               } else {
                  volGrid0->at(fsgrids::volfields::EXVOL) = 0.0;
               }

               if ( sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY ||
                    (sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY && sysBoundaryLayer == 1)
                  ) {
                  #ifdef DEBUG_FSOLVER
                  bool ok = true;
                  if (s.xRght == s.center) ok = false;
                  if (s.zRght == s.center) == NULL) ok = false;
                  if (s.xzTopRght < 0) ok = false;
                  if (ok == false) {
                     stringstream ss;
                     ss << "ERROR, got NULL neighbor in " << __FILE__ << ":" << __LINE__ << endl;
                     cerr << ss.str(); exit(1);
                  }
                  #endif

                  auto EGrid_i2j1k1 = &EData[s.xRght];
                  auto EGrid_i1j1k2 = &EData[s.zRght];
                  auto EGrid_i2j1k2 = &EData[s.xzTopRght];

                  CHECK_FLOAT(EGrid_i1j1k1->at(fsgrids::efield::EY));
                  CHECK_FLOAT(EGrid_i2j1k1->at(fsgrids::efield::EY));
                  CHECK_FLOAT(EGrid_i1j1k2->at(fsgrids::efield::EY));
                  CHECK_FLOAT(EGrid_i2j1k2->at(fsgrids::efield::EY));
                  volGrid0->at(fsgrids::volfields::EYVOL) = FOURTH*(EGrid_i1j1k1->at(fsgrids::efield::EY) + EGrid_i2j1k1->at(fsgrids::efield::EY) + EGrid_i1j1k2->at(fsgrids::efield::EY) + EGrid_i2j1k2->at(fsgrids::efield::EY));
                  CHECK_FLOAT(volGrid0->at(fsgrids::volfields::EYVOL));
               } else {
                  volGrid0->at(fsgrids::volfields::EYVOL) = 0.0;
               }

               if ( sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY ||
                    (sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY && sysBoundaryLayer == 1)
                  ) {
                  #ifdef DEBUG_FSOLVER
                  bool ok = true;
                  if (s.xRght == s.center) ok = false;
                  if (s.yRght == s.center) == NULL) ok = false;
                  if (s.xyTopRght < 0) ok = false;
                  if (ok == false) {
                     stringstream ss;
                     ss << "ERROR, got NULL neighbor in " << __FILE__ << ":" << __LINE__ << endl;
                     cerr << ss.str(); exit(1);
                  }
                  #endif

                  auto EGrid_i2j1k1 = &EData[s.xRght];
                  auto EGrid_i1j2k1 = &EData[s.yRght];
                  auto EGrid_i2j2k1 = &EData[s.xyTopRght];

                  CHECK_FLOAT(EGrid_i1j1k1->at(fsgrids::efield::EZ));
                  CHECK_FLOAT(EGrid_i2j1k1->at(fsgrids::efield::EZ));
                  CHECK_FLOAT(EGrid_i1j2k1->at(fsgrids::efield::EZ));
                  CHECK_FLOAT(EGrid_i2j2k1->at(fsgrids::efield::EZ));
                  volGrid0->at(fsgrids::volfields::EZVOL) = FOURTH*(EGrid_i1j1k1->at(fsgrids::efield::EZ) + EGrid_i2j1k1->at(fsgrids::efield::EZ) + EGrid_i1j2k1->at(fsgrids::efield::EZ) + EGrid_i2j2k1->at(fsgrids::efield::EZ));
                  CHECK_FLOAT(volGrid0->at(fsgrids::volfields::EZVOL));
               } else {
                  volGrid0->at(fsgrids::volfields::EZVOL) = 0.0;
               }
   });

   timer.stop();
}
