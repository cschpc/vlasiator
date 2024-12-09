
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
void calculateVolumeAveragedFieldsSimple(
    fsgrid::FsGrid<std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH>& perBGrid,
    fsgrid::FsGrid<std::array<Real, fsgrids::efield::N_EFIELD>, FS_STENCIL_WIDTH>& EGrid,
    fsgrid::FsGrid<std::array<Real, fsgrids::dperb::N_DPERB>, FS_STENCIL_WIDTH>& dPerBGrid,
    fsgrid::FsGrid<std::array<Real, fsgrids::volfields::N_VOL>, FS_STENCIL_WIDTH>& volGrid,
    fsgrid::FsGrid<fsgrids::technical, FS_STENCIL_WIDTH>& technicalGrid);

#endif
