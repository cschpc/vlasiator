
#ifndef LDZ_VOLUME
#define LDZ_VOLUME

#include <vector>

#include "fs_common.h"
//#include "../spatial_cells/spatial_cell_wrapper.hpp"

/*! \brief Top-level field averaging function.
 * 
 * Averages the electric and magnetic fields over the cell volumes.
 * 
 * \sa reconstructionCoefficients
 */
void calculateVolumeAveragedFieldsSimple(std::span<std::array<Real, fsgrids::bfield::N_BFIELD>> perb,
                                         std::span<std::array<Real, fsgrids::efield::N_EFIELD>> e,
                                         std::span<std::array<Real, fsgrids::dperb::N_DPERB>> dperb,
                                         std::span<std::array<Real, fsgrids::volfields::N_VOL>> vol,
                                         std::span<fsgrids::technical> technical, FieldSolverGrid &fsgrid);

#endif
