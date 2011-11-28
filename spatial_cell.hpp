/*!
Spatial cell class for Vlasiator that supports a variable number of velocity blocks.
*/

#ifndef VLASIATOR_SPATIAL_CELL_HPP
#define VLASIATOR_SPATIAL_CELL_HPP

#include "algorithm"
#include "boost/array.hpp"

#ifndef NO_SPARSE
#include "boost/unordered_map.hpp"
#include "boost/unordered_set.hpp"
#endif

#include "cmath"
#include "fstream"
#include "iostream"
#include "mpi.h"
#include "limits"
#include "stdint.h"
#include "vector"

#include "common.h"

namespace vlasiator {
namespace spatial_cell {


/**************************************
	User-modifiable part
**************************************/

// range of velocity grid in m/s
const double cell_vx_min = -1e6;
const double cell_vx_max = +1e6;
const double cell_vy_min = -1e6;
const double cell_vy_max = +1e6;
const double cell_vz_min = -1e6;
const double cell_vz_max = +1e6;

// length of a velocity block in velocity cells
#ifdef SPATIAL_CELL_BLOCK_LEN_X
const unsigned int block_len_x = SPATIAL_CELL_BLOCK_LEN_X;
#else
const unsigned int block_len_x = WID;
#endif
#ifdef SPATIAL_CELL_BLOCK_LEN_Y
const unsigned int block_len_y = SPATIAL_CELL_BLOCK_LEN_Y;
#else
const unsigned int block_len_y = WID;
#endif
#ifdef SPATIAL_CELL_BLOCK_LEN_Z
const unsigned int block_len_z = SPATIAL_CELL_BLOCK_LEN_Z;
#else
const unsigned int block_len_z = WID;
#endif

// lengths of spatial cells' velocity grid in velocity blocks
#ifdef SPATIAL_CELL_LEN_X
const unsigned int cell_len_x = SPATIAL_CELL_LEN_X;
#else
const unsigned int cell_len_x = 10;
#endif
#ifdef SPATIAL_CELL_LEN_Y
const unsigned int cell_len_y = SPATIAL_CELL_LEN_Y;
#else
const unsigned int cell_len_y = 10;
#endif
#ifdef SPATIAL_CELL_LEN_Z
const unsigned int cell_len_z = SPATIAL_CELL_LEN_Z;
#else
const unsigned int cell_len_z = 10;
#endif

/*******************************************
	End of user-modifiable part
*******************************************/

const double cell_dvx = (cell_vx_max - cell_vx_min) / (cell_len_x * block_len_x);
const double cell_dvy = (cell_vy_max - cell_vy_min) / (cell_len_y * block_len_y);
const double cell_dvz = (cell_vz_max - cell_vz_min) / (cell_len_z * block_len_z);

// constants for directions for example in neighbour lists
const unsigned int neg_x_dir = 0;
const unsigned int pos_x_dir = 1;
const unsigned int neg_y_dir = 2;
const unsigned int pos_y_dir = 3;
const unsigned int neg_z_dir = 4;
const unsigned int pos_z_dir = 5;

/*!
Used as an error from functions returning velocity cells or
as a cell that would be outside of the velocity block
*/
const unsigned int error_velocity_cell = std::numeric_limits<unsigned int>::max();

// only velocity cells that share a face are considered neighbors
const unsigned int n_neighbor_velocity_cells = 6;

/*!
Defines the indices of a velocity cell in a velocity block.
Indices start from 0 and the first value is the index in x direction.
*/
typedef boost::array<unsigned int, 3> velocity_cell_indices_t;

/*!
Used as an error from functions returning velocity cell indices or
as an index that would be outside of the velocity block
*/
const unsigned int error_velocity_cell_index = std::numeric_limits<unsigned int>::max();

const unsigned int velocity_block_len = block_len_x * block_len_y * block_len_z;

// only velocity blocks that share a face are considered neighbors
const unsigned int n_neighbor_velocity_blocks = 6;

class Velocity_Block {
public:
	// value of the distribution function
	double data[velocity_block_len];
	// spatial derivatives of the distribution function
	// TODO: #ifdef USE_KT_SOLVER for derivatives
	double d1x[SIZE_DERIV];
	double d2x[SIZE_DERIV];
	double d1y[SIZE_DERIV];
	double d2y[SIZE_DERIV];
	double d1z[SIZE_DERIV];
	double d2z[SIZE_DERIV];
	// spatial fluxes of this block
	double fx[SIZE_FLUXS];
	// TODO: #ifdef USE_KT_SOLVER for fy and fz
	double fy[SIZE_FLUXS];
	double fz[SIZE_FLUXS];
	double parameters[BlockParams::N_VELOCITY_BLOCK_PARAMS];
	Velocity_Block* neighbors[n_neighbor_velocity_blocks];

	/*!
	Sets data, derivatives and fluxes of this block to zero.
	*/
	void clear(void)
	{
		for (unsigned int i = 0; i < velocity_block_len; i++) {
			this->data[i] = 0;
		}
		for (unsigned int i = 0; i < SIZE_DERIV; i++) {
			this->d1x[i] = 0;
			this->d2x[i] = 0;
			this->d1y[i] = 0;
			this->d2y[i] = 0;
			this->d1z[i] = 0;
			this->d2z[i] = 0;
		}
		for (unsigned int i = 0; i < SIZE_FLUXS; i++) {
			this->fx[i] = 0;
			this->fy[i] = 0;
			this->fz[i] = 0;
		}
	}
};

/*!
Defines the indices of a velocity block in the velocity grid.
Indices start from 0 and the first value is the index in x direction.
*/
typedef boost::array<unsigned int, 3> velocity_block_indices_t;

/*!
Used as an error from functions returning velocity blocks indices or
as an index that would be outside of the velocity grid in this cell
*/
const unsigned int error_velocity_block_index = std::numeric_limits<unsigned int>::max();

/*!
Used as an error from functions returning velocity blocks or
as a block that would be outside of the velocity grid in this cell
*/
const unsigned int error_velocity_block = std::numeric_limits<unsigned int>::max();

const unsigned int max_velocity_blocks = cell_len_x * cell_len_y * cell_len_z;


const double cell_dx = (cell_vx_max - cell_vx_min) / cell_len_x;
const double cell_dy = (cell_vy_max - cell_vy_min) / cell_len_y;
const double cell_dz = (cell_vz_max - cell_vz_min) / cell_len_z;

// TODO: typedef unsigned int velocity_cell_t;
// TODO: typedef unsigned int velocity_block_t;

/****************************
 * Velocity block functions *
 ****************************/

/*!
Returns the indices of given velocity block
*/
velocity_block_indices_t get_velocity_block_indices(const unsigned int block) {
	velocity_block_indices_t indices;

	if (block >= max_velocity_blocks) {
		indices[0] = indices[1] = indices[2] = error_velocity_block_index;
	} else {
		indices[0] = block % cell_len_x;
		indices[1] = (block / cell_len_x) % cell_len_y;
		indices[2] = block / (cell_len_x * cell_len_y);
	}

	return indices;
}


/*!
Returns the velocity block at given indices or error_velocity_block
*/
unsigned int get_velocity_block(const velocity_block_indices_t indices) {
	if (indices[0] >= cell_len_x
	|| indices[1] >= cell_len_y
	|| indices[2] >= cell_len_z) {
		return error_velocity_block;
	}

	return indices[0] + indices[1] * cell_len_x + indices[2] * cell_len_x * cell_len_y;
}

/*!
Returns the velocity block at given location or
error_velocity_block if outside of the velocity grid
*/
unsigned int get_velocity_block(
	const double vx,
	const double vy,
	const double vz
)
{
	if (vx < cell_vx_min || vx >= cell_vx_max
	|| vy < cell_vy_min || vy >= cell_vy_max
	|| vz < cell_vz_min || vz >= cell_vz_max) {
		return error_velocity_block;
	}

	const velocity_block_indices_t indices = {
		(unsigned int) floor((vx - cell_vx_min) / cell_dx),
		(unsigned int) floor((vy - cell_vy_min) / cell_dy),
		(unsigned int) floor((vz - cell_vz_min) / cell_dz)
	};

	return get_velocity_block(indices);
}

/*!
Returns the id of a velocity block that is neighboring given block in given direction.
Returns error_velocity_block in case the neighboring velocity block would be outside
of the velocity grid.
*/
unsigned int get_velocity_block(
	const unsigned int block,
	const unsigned int direction
)
{
	const velocity_block_indices_t indices = get_velocity_block_indices(block);
	if (indices[0] == error_velocity_block_index) {
		return error_velocity_block;
	}

	// TODO: iterate over coordinates using a loop
	switch (direction) {
	case neg_x_dir:
		if (indices[0] == 0) {
			// TODO?: periodic velocity grid
			return error_velocity_block;
		} else {
			return block - 1;
		}
		break;

	case pos_x_dir:
		if (indices[0] >= cell_len_x - 1) {
			return error_velocity_block;
		} else {
			return block + 1;
		}
		break;

	case neg_y_dir:
		if (indices[1] == 0) {
			return error_velocity_block;
		} else {
			return block - cell_len_x;
		}
		break;

	case pos_y_dir:
		if (indices[1] >= cell_len_y - 1) {
			return error_velocity_block;
		} else {
			return block + cell_len_x;
		}
		break;

	case neg_z_dir:
		if (indices[2] == 0) {
			return error_velocity_block;
		} else {
			return block - cell_len_x * cell_len_y;
		}
		break;

	case pos_z_dir:
		if (indices[2] >= cell_len_z - 1) {
			return error_velocity_block;
		} else {
			return block + cell_len_x * cell_len_y;
		}
		break;

	default:
		return error_velocity_block;
		break;
	}
}


/*!
Returns the edge where given velocity block starts.
*/
double get_velocity_block_vx_min(const unsigned int block) {
	if (block == error_velocity_block) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	if (block >= max_velocity_blocks) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	const velocity_block_indices_t indices = get_velocity_block_indices(block);
	if (indices[0] == error_velocity_block_index) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	return cell_vx_min + cell_dx * indices[0];
}

/*!
Returns the edge where given velocity block ends.
*/
double get_velocity_block_vx_max(const unsigned int block) {
	return get_velocity_block_vx_min(block) + cell_dx;
}


/*!
Returns the edge where given velocity block starts.
*/
double get_velocity_block_vy_min(const unsigned int block) {
	if (block == error_velocity_block) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	if (block >= max_velocity_blocks) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	const velocity_block_indices_t indices = get_velocity_block_indices(block);
	if (indices[1] == error_velocity_block_index) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	return cell_vy_min + cell_dy * indices[1];
}

/*!
Returns the edge where given velocity block ends.
*/
double get_velocity_block_vy_max(const unsigned int block) {
	return get_velocity_block_vy_min(block) + cell_dy;
}


/*!
Returns the edge where given velocity block starts.
*/
double get_velocity_block_vz_min(const unsigned int block) {
	if (block == error_velocity_block) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	if (block >= max_velocity_blocks) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	const velocity_block_indices_t indices = get_velocity_block_indices(block);
	if (indices[2] == error_velocity_block_index) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	return cell_vz_min + cell_dz * indices[2];
}

/*!
Returns the edge where given velocity block ends.
*/
double get_velocity_block_vz_max(const unsigned int block) {
	return get_velocity_block_vz_min(block) + cell_dz;
}


/***************************
 * Velocity cell functions *
 ***************************/

/*!
Returns the indices of given velocity cell
*/
velocity_cell_indices_t get_velocity_cell_indices(const unsigned int cell) {
	velocity_cell_indices_t indices;

	if (cell >= velocity_block_len) {
		indices[0] = indices[1] = indices[2] = error_velocity_cell_index;
	} else {
		indices[0] = cell % block_len_x;
		indices[1] = (cell / block_len_x) % block_len_y;
		indices[2] = cell / (block_len_x * block_len_y);
	}

	return indices;
}

/*!
Returns the velocity cell at given indices or error_velocity_cell
*/
unsigned int get_velocity_cell(const velocity_cell_indices_t indices) {
	if (indices[0] >= block_len_x
	|| indices[1] >= block_len_y
	|| indices[2] >= block_len_z) {
		return error_velocity_cell;
	}
	return indices[0] + indices[1] * block_len_x + indices[2] * block_len_x * block_len_y;
}


/*!
Returns the id of a velocity cell that is neighboring given cell in given direction.
Returns error_velocity_cell in case the neighboring velocity cell would be outside
of the velocity block.
*/
unsigned int get_velocity_cell(
	const unsigned int cell,
	const unsigned int direction
)
{
	const velocity_cell_indices_t indices = get_velocity_cell_indices(cell);
	if (indices[0] == error_velocity_cell_index) {
		return error_velocity_cell;
	}

	// TODO: iterate over coordinates using a loop
	switch (direction) {

	case neg_x_dir:
		if (indices[0] == 0) {
			// TODO?: periodic velocity grid
			return error_velocity_cell;
		} else {
			return cell - 1;
		}
		break;

	case pos_x_dir:
		if (indices[0] >= block_len_x - 1) {
			return error_velocity_cell;
		} else {
			return cell + 1;
		}
		break;

	case neg_y_dir:
		if (indices[1] == 0) {
			return error_velocity_cell;
		} else {
			return cell - block_len_x;
		}
		break;

	case pos_y_dir:
		if (indices[1] >= block_len_y - 1) {
			return error_velocity_cell;
		} else {
			return cell + block_len_x;
		}
		break;

	case neg_z_dir:
		if (indices[2] == 0) {
			return error_velocity_cell;
		} else {
			return cell - block_len_x * block_len_y;
		}
		break;

	case pos_z_dir:
		if (indices[2] >= block_len_z - 1) {
			return error_velocity_cell;
		} else {
			return cell + block_len_x * block_len_y;
		}
		break;

	default:
		return error_velocity_cell;
		break;
	}
}


/*!
Returns the velocity cell at given location or
error_velocity_cell if outside of given velocity block.
*/
unsigned int get_velocity_cell(
	const unsigned int velocity_block,
	const double vx,
	const double vy,
	const double vz
)
{
	const double block_vx_min = get_velocity_block_vx_min(velocity_block);
	const double block_vx_max = get_velocity_block_vx_max(velocity_block);
	const double block_vy_min = get_velocity_block_vy_min(velocity_block);
	const double block_vy_max = get_velocity_block_vy_max(velocity_block);
	const double block_vz_min = get_velocity_block_vz_min(velocity_block);
	const double block_vz_max = get_velocity_block_vz_max(velocity_block);

	if (vx < block_vx_min || vx >= block_vx_max
	|| vy < block_vy_min || vy >= block_vy_max
	|| vz < block_vz_min || vz >= block_vz_max
	) {
		return error_velocity_cell;
	}

	const velocity_block_indices_t indices = {
		(unsigned int) floor((vx - block_vx_min) / ((block_vx_max - block_vx_min) / block_len_x)),
		(unsigned int) floor((vy - block_vy_min) / ((block_vy_max - block_vy_min) / block_len_y)),
		(unsigned int) floor((vz - block_vz_min) / ((block_vz_max - block_vz_min) / block_len_z))
	};

	return get_velocity_cell(indices);
}


/*!
Returns the edge where given velocity cell in the given velocity block starts.
TODO: move these to velocity cell class?
*/
double get_velocity_cell_vx_min(
	const unsigned int velocity_block,
	const unsigned int velocity_cell
)
{
	if (velocity_cell == error_velocity_cell) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	const velocity_cell_indices_t indices = get_velocity_cell_indices(velocity_cell);
	if (indices[0] == error_velocity_cell_index) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	const double block_vx_min = get_velocity_block_vx_min(velocity_block);
	const double block_vx_max = get_velocity_block_vx_max(velocity_block);

	return block_vx_min + (block_vx_max - block_vx_min) / block_len_x * indices[0];
}

/*!
Returns the edge where given velocity cell in the given velocity block ends.
*/
double get_velocity_cell_vx_max(
	const unsigned int velocity_block,
	const unsigned int velocity_cell
)
{
	if (velocity_cell == error_velocity_cell) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	const velocity_cell_indices_t indices = get_velocity_cell_indices(velocity_cell);
	if (indices[0] == error_velocity_cell_index) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	const double block_vx_min = get_velocity_block_vx_min(velocity_block);
	const double block_vx_max = get_velocity_block_vx_max(velocity_block);

	return block_vx_min + (block_vx_max - block_vx_min) / block_len_x * (indices[0] + 1);
}

/*!
Returns the edge where given velocity cell in the given velocity block starts.
*/
double get_velocity_cell_vy_min(
	const unsigned int velocity_block,
	const unsigned int velocity_cell
)
{
	if (velocity_cell == error_velocity_cell) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	const velocity_cell_indices_t indices = get_velocity_cell_indices(velocity_cell);
	if (indices[1] == error_velocity_cell_index) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	const double block_vy_min = get_velocity_block_vy_min(velocity_block);
	const double block_vy_max = get_velocity_block_vy_max(velocity_block);

	return block_vy_min + (block_vy_max - block_vy_min) / block_len_y * indices[1];
}

/*!
Returns the edge where given velocity cell in the given velocity block ends.
*/
double get_velocity_cell_vy_max(
	const unsigned int velocity_block,
	const unsigned int velocity_cell
)
{
	if (velocity_cell == error_velocity_cell) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	const velocity_cell_indices_t indices = get_velocity_cell_indices(velocity_cell);
	if (indices[1] == error_velocity_cell_index) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	const double block_vy_min = get_velocity_block_vy_min(velocity_block);
	const double block_vy_max = get_velocity_block_vy_max(velocity_block);

	return block_vy_min + (block_vy_max - block_vy_min) / block_len_y * (indices[1] + 1);
}

/*!
Returns the edge where given velocity cell in the given velocity block starts.
*/
double get_velocity_cell_vz_min(
	const unsigned int velocity_block,
	const unsigned int velocity_cell
)
{
	if (velocity_cell == error_velocity_cell) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	const velocity_cell_indices_t indices = get_velocity_cell_indices(velocity_cell);
	if (indices[2] == error_velocity_cell_index) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	const double block_vz_min = get_velocity_block_vz_min(velocity_block);
	const double block_vz_max = get_velocity_block_vz_max(velocity_block);

	return block_vz_min + (block_vz_max - block_vz_min) / block_len_z * indices[2];
}

/*!
Returns the edge where given velocity cell in the given velocity block ends.
*/
double get_velocity_cell_vz_max(
	const unsigned int velocity_block,
	const unsigned int velocity_cell
)
{
	if (velocity_cell == error_velocity_cell) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	const velocity_cell_indices_t indices = get_velocity_cell_indices(velocity_cell);
	if (indices[2] == error_velocity_cell_index) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	const double block_vz_min = get_velocity_block_vz_min(velocity_block);
	const double block_vz_max = get_velocity_block_vz_max(velocity_block);

	return block_vz_min + (block_vz_max - block_vz_min) / block_len_z * (indices[2] + 1);
}


class Spatial_Cell {
public:

	Spatial_Cell()
	{
		/*
		Block list always has room for all blocks
		*/
		this->velocity_block_list.reserve(max_velocity_blocks);

		#ifdef NO_SPARSE
		this->velocity_blocks.resize(max_velocity_blocks);
		for (unsigned int block = 0; block < max_velocity_blocks; block++) {
			this->velocity_block_list.push_back(block);
		}
		#else
		this->block_address_cache.reserve(max_velocity_blocks);
		for (unsigned int block = 0; block < max_velocity_blocks; block++) {
			this->velocity_block_list.push_back(error_velocity_block);
			this->block_address_cache.push_back(&(this->null_block));
		}
		#endif

		this->null_block.clear();
		// zero neighbor lists of null block
		for (unsigned int i = 0; i < n_neighbor_velocity_blocks; i++) {
			this->null_block.neighbors[i] = NULL;
		}

		this->velocity_block_min_value = 0;
		this->velocity_block_min_avg_value = 0;

		// add spatial cell parameters
		this->parameters.reserve(CellParams::N_SPATIAL_CELL_PARAMS);
		for (unsigned int i = 0; i < CellParams::N_SPATIAL_CELL_PARAMS; i++) {
			this->parameters.push_back(0);
		}
	}


	/*!
	Returns a reference to the given velocity block or to
	the null block if given velocity block doesn't exist.
	*/
	Velocity_Block& at(const unsigned int block)
	{
		if (block == error_velocity_block
		|| block >= max_velocity_blocks) {
			return this->null_block;
		} else {
			#ifdef NO_SPARSE
			return this->velocity_blocks.at(block);
			#else
			return *(this->block_address_cache.at(block));
			#endif
		}
	}

	/*!
	A const version of the non-const at function.
	*/
	Velocity_Block const& at(const unsigned int block) const
	{
		if (block == error_velocity_block
		|| block >= max_velocity_blocks) {
			return this->null_block;
		} else {
			#ifdef NO_SPARSE
			return this->velocity_blocks.at(block);
			#else
			return *(this->block_address_cache.at(block));
			#endif
		}
	}


	/*!
	Returns the number of given velocity blocks that exist.
	*/
	size_t count(const unsigned int block) const
	{
		#ifdef NO_SPARSE
		if (block == error_velocity_block
		|| block >= max_velocity_blocks) {
			return 0;
		} else {
			return 1;
		}
		#else
		return this->velocity_blocks.count(block);
		#endif
	}


	/*!
	Returns the number of existing velocity blocks.
	*/
	size_t size(void) const
	{
		return this->velocity_blocks.size();
	}


	/*!
	Sets the given value to a velocity cell at given coordinates.

	Creates the velocity block at given coordinates if it doesn't exist.
	*/
	void set_value(const double vx, const double vy, const double vz, const double value)
	{
		const unsigned int block = get_velocity_block(vx, vy, vz);
		#ifndef NO_SPARSE
		if (this->velocity_blocks.count(block) == 0) {
			if (!this->add_velocity_block(block)) {
				std::cerr << "Couldn't add velocity block " << block << std::endl;
				abort();
			}
		}
		#endif
		std::cout << "Added block " << block << " in set_value" << std::endl;

		Velocity_Block* block_ptr = &(this->velocity_blocks.at(block));

		const unsigned int cell = get_velocity_cell(block, vx, vy, vz);
		block_ptr->data[cell] = value;
	}


	void* at(void)
	{
		return this;
	}


	MPI_Datatype mpi_datatype(void)
	{
		MPI_Datatype type;
		std::vector<MPI_Aint> displacements;
		std::vector<int> block_lengths;
		unsigned int block_index = 0;

		switch (Spatial_Cell::mpi_transfer_type) {
		case 0:
			MPI_Type_contiguous(0, MPI_BYTE, &type);
			break;

		// send velocity block list
		case 1:
			displacements.push_back((uint8_t*) &(this->velocity_block_list[0]) - (uint8_t*) this);
			block_lengths.push_back(sizeof(unsigned int) * max_velocity_blocks);

			MPI_Type_create_hindexed(
				displacements.size(),
				&block_lengths[0],
				&displacements[0],
				MPI_BYTE,
				&type
			);
			break;

		// send velocity block data
		case 2:
			displacements.reserve(this->velocity_blocks.size());
			block_lengths.reserve(this->velocity_blocks.size());

			while (block_index < max_velocity_blocks
			&& this->velocity_block_list[block_index] != error_velocity_block) {

				// debug
				#ifndef NO_SPARSE
				if (this->velocity_blocks.count(this->velocity_block_list[block_index]) == 0) {
					int proc;
					MPI_Comm_rank(MPI_COMM_WORLD, &proc);
					std::cerr << __FILE__ << ":" << __LINE__
						<< " Process " << proc
						<< ": Velocity block " << this->velocity_block_list[block_index]
						<< " doesn't exist at index " << block_index
						<< std::endl;
					abort();
				}
				#endif

				// TODO: use cached block addresses
				displacements.push_back((uint8_t*) this->velocity_blocks.at(this->velocity_block_list[block_index]).data - (uint8_t*) this);
				block_lengths.push_back(sizeof(double) * velocity_block_len);

				block_index++;
			}

			if (displacements.size() > 0) {
				MPI_Type_create_hindexed(
					displacements.size(),
					&block_lengths[0],
					&displacements[0],
					MPI_BYTE,
					&type
				);
			} else {
				MPI_Type_contiguous(0, MPI_BYTE, &type);
			}
			break;

		// send velocity block fluxes
		case 3:
			displacements.reserve(this->velocity_blocks.size());
			block_lengths.reserve(this->velocity_blocks.size());

			while (block_index < max_velocity_blocks
			&& this->velocity_block_list[block_index] != error_velocity_block) {

				// debug
				#ifndef NO_SPARSE
				if (this->velocity_blocks.count(this->velocity_block_list[block_index]) == 0) {
					int proc;
					MPI_Comm_rank(MPI_COMM_WORLD, &proc);
					std::cerr << __FILE__ << ":" << __LINE__
						<< " Process " << proc
						<< ": Velocity block " << this->velocity_block_list[block_index]
						<< " doesn't exist at index " << block_index
						<< std::endl;
					abort();
				}
				#endif

				displacements.push_back((uint8_t*) this->velocity_blocks.at(this->velocity_block_list[block_index]).fx - (uint8_t*) this);
				block_lengths.push_back(sizeof(double) * 3 * SIZE_FLUXS);

				block_index++;
			}

			if (displacements.size() > 0) {
				MPI_Type_create_hindexed(
					displacements.size(),
					&block_lengths[0],
					&displacements[0],
					MPI_BYTE,
					&type
				);
			} else {
				MPI_Type_contiguous(0, MPI_BYTE, &type);
			}
			break;

		// send spatial cell parameters
		case 4:
			displacements.push_back((uint8_t*) &(this->velocity_block_min_value) - (uint8_t*) this);
			block_lengths.push_back(sizeof(double));

			displacements.push_back((uint8_t*) &(this->velocity_block_min_avg_value) - (uint8_t*) this);
			block_lengths.push_back(sizeof(double));

			displacements.push_back((uint8_t*) &(this->parameters[0]) - (uint8_t*) this);
			block_lengths.push_back(sizeof(double) * CellParams::N_SPATIAL_CELL_PARAMS);

			MPI_Type_create_hindexed(
				displacements.size(),
				&block_lengths[0],
				&displacements[0],
				MPI_BYTE,
				&type
			);
			break;

		// send velocity block derivatives
		case 5:
			displacements.reserve(this->velocity_blocks.size());
			block_lengths.reserve(this->velocity_blocks.size());

			while (block_index < max_velocity_blocks
			&& this->velocity_block_list[block_index] != error_velocity_block) {

				// debug
				#ifndef NO_SPARSE
				if (this->velocity_blocks.count(this->velocity_block_list[block_index]) == 0) {
					int proc;
					MPI_Comm_rank(MPI_COMM_WORLD, &proc);
					std::cerr << __FILE__ << ":" << __LINE__
						<< " Process " << proc
						<< ": Velocity block " << this->velocity_block_list[block_index]
						<< " doesn't exist at index " << block_index
						<< std::endl;
					abort();
				}
				#endif

				displacements.push_back((uint8_t*) this->velocity_blocks.at(this->velocity_block_list[block_index]).d1x - (uint8_t*) this);
				block_lengths.push_back(sizeof(double) * 6 * SIZE_DERIV);

				block_index++;
			}

			if (displacements.size() > 0) {
				MPI_Type_create_hindexed(
					displacements.size(),
					&block_lengths[0],
					&displacements[0],
					MPI_BYTE,
					&type
				);
			} else {
				MPI_Type_contiguous(0, MPI_BYTE, &type);
			}
			break;

		// send spatial cell parameters and velocity block data, derivatives, fluxes and parameters
		case 6:
			displacements.reserve(3 + this->velocity_blocks.size());
			block_lengths.reserve(3 + this->velocity_blocks.size());

			displacements.push_back((uint8_t*) &(this->velocity_block_min_value) - (uint8_t*) this);
			block_lengths.push_back(sizeof(double));

			displacements.push_back((uint8_t*) &(this->velocity_block_min_avg_value) - (uint8_t*) this);
			block_lengths.push_back(sizeof(double));

			displacements.push_back((uint8_t*) &(this->parameters[0]) - (uint8_t*) this);
			block_lengths.push_back(sizeof(double) * CellParams::N_SPATIAL_CELL_PARAMS);

			while (block_index < max_velocity_blocks
			&& this->velocity_block_list[block_index] != error_velocity_block) {

				// debug
				#ifndef NO_SPARSE
				if (this->velocity_blocks.count(this->velocity_block_list[block_index]) == 0) {
					int proc;
					MPI_Comm_rank(MPI_COMM_WORLD, &proc);
					std::cerr << __FILE__ << ":" << __LINE__
						<< " Process " << proc
						<< ": Velocity block " << this->velocity_block_list[block_index]
						<< " doesn't exist at index " << block_index
						<< std::endl;
					abort();
				}
				#endif

				displacements.push_back((uint8_t*) this->velocity_blocks.at(this->velocity_block_list[block_index]).data - (uint8_t*) this);
				block_lengths.push_back(sizeof(double) * 10 * velocity_block_len + sizeof(double) * BlockParams::N_VELOCITY_BLOCK_PARAMS);

				block_index++;
			}

			if (displacements.size() > 0) {
				MPI_Type_create_hindexed(
					displacements.size(),
					&block_lengths[0],
					&displacements[0],
					MPI_BYTE,
					&type
				);
			} else {
				MPI_Type_contiguous(0, MPI_BYTE, &type);
			}
			break;

		// send velocity block parameters
		case 7:
			displacements.reserve(this->velocity_blocks.size());
			block_lengths.reserve(this->velocity_blocks.size());

			while (block_index < max_velocity_blocks
			&& this->velocity_block_list[block_index] != error_velocity_block) {

				// debug
				#ifndef NO_SPARSE
				if (this->velocity_blocks.count(this->velocity_block_list[block_index]) == 0) {
					int proc;
					MPI_Comm_rank(MPI_COMM_WORLD, &proc);
					std::cerr << __FILE__ << ":" << __LINE__
						<< " Process " << proc
						<< ": Velocity block " << this->velocity_block_list[block_index]
						<< " doesn't exist at index " << block_index
						<< std::endl;
					abort();
				}
				#endif

				// TODO: use cached block addresses
				displacements.push_back((uint8_t*) this->velocity_blocks.at(this->velocity_block_list[block_index]).parameters - (uint8_t*) this);
				block_lengths.push_back(sizeof(double) * BlockParams::N_VELOCITY_BLOCK_PARAMS);

				block_index++;
			}

			if (displacements.size() > 0) {
				MPI_Type_create_hindexed(
					displacements.size(),
					&block_lengths[0],
					&displacements[0],
					MPI_BYTE,
					&type
				);
			} else {
				MPI_Type_contiguous(0, MPI_BYTE, &type);
			}
			break;

		default:
			std::cerr << __FILE__ << ":" << __LINE__ << " Unsupported mpi transfer type." << std::endl;
			abort();
			break;
		}

		return type;
	}


	/*!
	Sets the minimum velocity cell value of a distrubution function for
	that velocity block to be considered to have contents.
	*/
	void set_block_minimum(const double value)
	{
		this->velocity_block_min_value = value;
	}

	/*!
	Sets the minimum average velocity cell value of a distrubution function
	within a block for that block to be considered to have contents.
	*/
	void set_block_average_minimum(const double value)
	{
		this->velocity_block_min_avg_value = value;
	}

	/*!
	Returns true if given velocity block has enough of a distribution function.
	Returns false if the value of the distribution function is too low in every
	sense in given block.
	Also returns false if given block doesn't exist or is an error block.
	*/
	bool velocity_block_has_contents(
		#ifdef NO_SPARSE
		const unsigned int /*block*/
		#else
		const unsigned int block
		#endif
	) const
	{
		#ifndef NO_SPARSE
		if (block == error_velocity_block
		|| this->velocity_blocks.count(block) == 0) {
			return false;
		}

		bool has_content = false;

		double total = 0;
		const Velocity_Block* block_ptr = &(this->velocity_blocks.at(block));

		for (unsigned int i = 0; i < velocity_block_len; i++) {
			total += block_ptr->data[i];
			if (block_ptr->data[i] >= this->velocity_block_min_value) {
				has_content = true;
				break;
			}
		}

		if (total >= this->velocity_block_min_avg_value * velocity_block_len) {
			has_content = true;
		}

		return has_content;
		#else
		return true;
		#endif
	}


	/*!
	Returns the total value of the distribution function within this spatial cell.
	*/
	double get_total_value(void) const
	{
		double total = 0;

		for (auto block = this->velocity_blocks.cbegin(); block != this->velocity_blocks.cend(); block++) {
			for (unsigned int i = 0; i < velocity_block_len; i++) {
				#ifdef NO_SPARSE
				total += block->data[i];
				#else
				total += block->second.data[i];
				#endif
			}
		}

		return total;
	}


	/*!
	Returns the total size of the data in this spatial cell in bytes.

	Does not include velocity block lists, the null velocity block or velocity block neighbor lists.
	*/
	size_t get_data_size(void) const
	{
		const unsigned int n = this->velocity_blocks.size();

		return 2 * sizeof(double)
			+ n * sizeof(unsigned int)
			+ n * 2 * velocity_block_len * sizeof(double);
	}


	/*!
	Saves this spatial cell in binary format into the given filename.

	Fileformat, native endian:
	1 double velocity_block_min_value
	1 double velocity_block_min_avg_value
	1 unsigned int random velocity block from the velocity grid
	N double where N == velocity_block_len, velocity block distribution function data
	N double velocity block flux data
	*/
	/*bool save_bin(const char* filename)
	{
		...
	}*/


	/*!
	Checks velocity blocks in the velocity block list.
	*/
	void check_velocity_block_list(void) const
	{
		for (unsigned int i = 0; i < max_velocity_blocks; i++) {
			if (this->velocity_block_list[i] == error_velocity_block) {
				for (unsigned int j = i; j < max_velocity_blocks; j++) {
					if (this->velocity_block_list[i] != error_velocity_block) {
						std::cerr << __FILE__ << ":" << __LINE__
							<< "Velocity block list has holes"
							<< std::endl;
						abort();
					}
				}
				break;
			}

			#ifndef NO_SPARSE
			if (this->velocity_blocks.count(this->velocity_block_list[i]) == 0) {
				std::cerr << __FILE__ << ":" << __LINE__
					<< " Velocity block " << this->velocity_block_list[i]
					<< " doesn't exist"
					<< std::endl;
				abort();
			}
			#endif
		}
	}

	/*!
	Prints velocity blocks in the velocity block list.
	*/
	void print_velocity_block_list(void) const
	{
		std::cout << this->velocity_blocks.size() << " blocks: ";
		for (unsigned int i = 0; i < max_velocity_blocks; i++) {

			if (this->velocity_block_list[i] == error_velocity_block) {
				// debug
				for (unsigned int j = i; j < max_velocity_blocks; j++) {
					if (this->velocity_block_list[i] != error_velocity_block) {
						std::cerr << "Velocity block list has holes" << std::endl;
						abort();
					}
				}
				break;
			}
			std::cout << this->velocity_block_list[i] << " ";
		}
		std::cout << std::endl;
	}

	/*!
	Prints given velocity block's velocity neighbor list.
	*/
	void print_velocity_neighbor_list(const unsigned int block) const
	{
		#ifndef NO_SPARSE
		if (this->velocity_blocks.count(block) == 0) {
			return;
		}
		#endif

		const Velocity_Block* block_ptr = &(this->velocity_blocks.at(block));

		std::cout << block << " neighbors: ";
		for (unsigned int neighbor = 0; neighbor < n_neighbor_velocity_blocks; neighbor++) {
				if (block_ptr->neighbors[neighbor] == NULL) {
				std::cout << "NULL ";
			} else if (block_ptr->neighbors[neighbor] == &(this->null_block)) {
				std::cout << "null ";
			} else {
				std::cout << block_ptr->neighbors[neighbor] << " ";
			}
		}
		std::cout << std::endl;
	}

	/*!
	Prints all velocity blocks' velocity neighbor list.
	*/
	void print_velocity_neighbor_lists(void) const
	{
		for (unsigned int i = 0; i < max_velocity_blocks; i++) {

			if (this->velocity_block_list[i] == error_velocity_block) {
				break;
			}

			this->print_velocity_neighbor_list(this->velocity_block_list[i]);
		}
	}

	/*!
	Adds "important" and removes "unimportant" velocity blocks to/from this cell.

	Removes all velocity blocks from this spatial cell which don't have content
	and don't have spatial or velocity neighbors with content.
	Adds neighbors for all velocity blocks which do have content (including spatial neighbors).
	Assumes that only blocks with a shared face in velocity space are neighbors.
	All cells in spatial_neighbors are assumed to be neighbors of this cell.
	*/
	void adjust_velocity_blocks(
		#ifdef NO_SPARSE
		const std::vector<Spatial_Cell*>& /*spatial_neighbors*/
		#else
		const std::vector<Spatial_Cell*>& spatial_neighbors
		#endif
	) {
		// debug
		this->check_velocity_block_list();

		#ifndef NO_SPARSE
		// don't iterate over blocks created / removed by this function
		std::vector<unsigned int> original_block_list;
		for (
			unsigned int block = this->velocity_block_list[0], block_i = 0;
			block_i < max_velocity_blocks
				&& this->velocity_block_list[block_i] != error_velocity_block;
			block = this->velocity_block_list[++block_i]
		) {
			original_block_list.push_back(block);
		}

		// get all velocity blocks with content in neighboring spatial cells
		boost::unordered_set<unsigned int> neighbors_with_content;
		for (std::vector<Spatial_Cell*>::const_iterator
			neighbor = spatial_neighbors.begin();
			neighbor != spatial_neighbors.end();
			neighbor++
		) {

			for (std::vector<unsigned int>::const_iterator
				block = (*neighbor)->velocity_block_list.begin();
				block != (*neighbor)->velocity_block_list.end();
				block++
			) {
				if (*block == error_velocity_block) {
					break;
				}

				if ((*neighbor)->velocity_block_has_contents(*block)) {
					neighbors_with_content.insert(*block);
				}
			}
		}

		// remove all local blocks without content and without neighbors with content
		for (std::vector<unsigned int>::const_iterator
			original = original_block_list.begin();
			original != original_block_list.end();
			original++
		) {
			const bool original_has_content = this->velocity_block_has_contents(*original);

			if (original_has_content) {

				// add missing neighbors in velocity space
				for (unsigned int direction = neg_x_dir; direction <= pos_z_dir; direction++) {
					const unsigned int neighbor_block = get_velocity_block(*original, direction);
					if (neighbor_block == error_velocity_block) {
						continue;
					}

					if (this->velocity_blocks.count(neighbor_block) > 0) {
						continue;
					}

					if (!this->add_velocity_block(neighbor_block)) {
						std::cerr << __FILE__ << ":" << __LINE__
							<< " Failed to add neighbor block " << neighbor_block
							<< " for block " << *original
							<< std::endl;
						abort();
					}
				}

			} else {

				// check if any neighbour has contents
				bool velocity_neighbors_have_content = false;
				for (unsigned int direction = neg_x_dir; direction <= pos_z_dir; direction++) {
					const unsigned int neighbor_block = get_velocity_block(*original, direction);
					if (this->velocity_block_has_contents(neighbor_block)) {
						velocity_neighbors_have_content = true;
						break;
					}
				}

				if (!velocity_neighbors_have_content
				&& neighbors_with_content.count(*original) == 0) {
					this->remove_velocity_block(*original);
				}
			}
		}

		// add local blocks for spatial neighbors with content
		for (boost::unordered_set<unsigned int>::const_iterator
			neighbor = neighbors_with_content.begin();
			neighbor != neighbors_with_content.end();
			neighbor++
		) {
			this->add_velocity_block(*neighbor);
		}
		#endif
	}

	/*!
	Saves this spatial cell in vtk ascii format into the given filename.
	*/
	void save_vtk(const char* filename) const {

		// do nothing if one cell or block dimension is 0
		if (block_len_x == 0
		|| block_len_y == 0
		|| block_len_z == 0
		|| cell_len_x == 0
		|| cell_len_y == 0
		|| cell_len_z == 0) {
			return;
		}

		std::ofstream outfile(filename);
		if (!outfile.is_open()) {
			std::cerr << "Couldn't open file " << filename << std::endl;
			// TODO: throw an exception instead
			abort();
		}

		outfile << "# vtk DataFile Version 2.0" << std::endl;
		outfile << "Vlasiator spatial cell" << std::endl;
		outfile << "ASCII" << std::endl;
		outfile << "DATASET UNSTRUCTURED_GRID" << std::endl;

		// write separate points for every velocity cells' corners
		outfile << "POINTS " << (this->velocity_blocks.size() * velocity_block_len + 2) * 8 << " double" << std::endl;
		for (std::vector<unsigned int>::const_iterator
			block = this->velocity_block_list.begin();
			block != this->velocity_block_list.end();
			block++
		) {

			if (*block == error_velocity_block) {
				// assume no blocks after first error block
				break;
			}

			for (unsigned int z_index = 0; z_index < block_len_z; z_index++)
			for (unsigned int y_index = 0; y_index < block_len_y; y_index++)
			for (unsigned int x_index = 0; x_index < block_len_x; x_index++) {

				const velocity_cell_indices_t indices = {x_index, y_index, z_index};
				const unsigned int velocity_cell = get_velocity_cell(indices);

				outfile << get_velocity_cell_vx_min(*block, velocity_cell) << " "
					<< get_velocity_cell_vy_min(*block, velocity_cell) << " "
					<< get_velocity_cell_vz_min(*block, velocity_cell) << std::endl;

				outfile << get_velocity_cell_vx_max(*block, velocity_cell) << " "
					<< get_velocity_cell_vy_min(*block, velocity_cell) << " "
					<< get_velocity_cell_vz_min(*block, velocity_cell) << std::endl;

				outfile << get_velocity_cell_vx_min(*block, velocity_cell) << " "
					<< get_velocity_cell_vy_max(*block, velocity_cell) << " "
					<< get_velocity_cell_vz_min(*block, velocity_cell) << std::endl;

				outfile << get_velocity_cell_vx_max(*block, velocity_cell) << " "
					<< get_velocity_cell_vy_max(*block, velocity_cell) << " "
					<< get_velocity_cell_vz_min(*block, velocity_cell) << std::endl;

				outfile << get_velocity_cell_vx_min(*block, velocity_cell) << " "
					<< get_velocity_cell_vy_min(*block, velocity_cell) << " "
					<< get_velocity_cell_vz_max(*block, velocity_cell) << std::endl;

				outfile << get_velocity_cell_vx_max(*block, velocity_cell) << " "
					<< get_velocity_cell_vy_min(*block, velocity_cell) << " "
					<< get_velocity_cell_vz_max(*block, velocity_cell) << std::endl;

				outfile << get_velocity_cell_vx_min(*block, velocity_cell) << " "
					<< get_velocity_cell_vy_max(*block, velocity_cell) << " "
					<< get_velocity_cell_vz_max(*block, velocity_cell) << std::endl;

				outfile << get_velocity_cell_vx_max(*block, velocity_cell) << " "
					<< get_velocity_cell_vy_max(*block, velocity_cell) << " "
					<< get_velocity_cell_vz_max(*block, velocity_cell) << std::endl;
			}
		}
		/*
		Add small velocity cells to the negative and positive corners of the grid
		so VisIt knows the maximum size of the velocity grid regardless of existing cells
		*/
		outfile << cell_vx_min - 0.1 * cell_dvx << " "
			<< cell_vy_min - 0.1 * cell_dvy << " "
			<< cell_vz_min - 0.1 * cell_dvz << std::endl;
		outfile << cell_vx_min << " "
			<< cell_vy_min - 0.1 * cell_dvy << " "
			<< cell_vz_min - 0.1 * cell_dvz << std::endl;
		outfile << cell_vx_min - 0.1 * cell_dvx << " "
			<< cell_vy_min << " "
			<< cell_vz_min - 0.1 * cell_dvz << std::endl;
		outfile << cell_vx_min << " "
			<< cell_vy_min << " "
			<< cell_vz_min - 0.1 * cell_dvz << std::endl;
		outfile << cell_vx_min - 0.1 * cell_dvx << " "
			<< cell_vy_min - 0.1 * cell_dvy << " "
			<< cell_vz_min << std::endl;
		outfile << cell_vx_min << " "
			<< cell_vy_min - 0.1 * cell_dvy << " "
			<< cell_vz_min << std::endl;
		outfile << cell_vx_min - 0.1 * cell_dvx << " "
			<< cell_vy_min << " "
			<< cell_vz_min << std::endl;
		outfile << cell_vx_min << " "
			<< cell_vy_min << " "
			<< cell_vz_min << std::endl;

		outfile << cell_vx_max << " "
			<< cell_vy_max << " "
			<< cell_vz_max << std::endl;
		outfile << cell_vx_max + 0.1 * cell_dvx << " "
			<< cell_vy_max << " "
			<< cell_vz_max << std::endl;
		outfile << cell_vx_max << " "
			<< cell_vy_max + 0.1 * cell_dvy << " "
			<< cell_vz_max << std::endl;
		outfile << cell_vx_max + 0.1 * cell_dvx << " "
			<< cell_vy_max + 0.1 * cell_dvy << " "
			<< cell_vz_max << std::endl;
		outfile << cell_vx_max << " "
			<< cell_vy_max << " "
			<< cell_vz_max + 0.1 * cell_dvz << std::endl;
		outfile << cell_vx_max + 0.1 * cell_dvx << " "
			<< cell_vy_max << " "
			<< cell_vz_max + 0.1 *  cell_dvz << std::endl;
		outfile << cell_vx_max << " "
			<< cell_vy_max + 0.1 * cell_dvy << " "
			<< cell_vz_max + 0.1 * cell_dvz << std::endl;
		outfile << cell_vx_max + 0.1 * cell_dvx << " "
			<< cell_vy_max + 0.1 * cell_dvy << " "
			<< cell_vz_max + 0.1 * cell_dvz << std::endl;

		// map cells to written points
		outfile << "CELLS "
			<< this->velocity_blocks.size() * velocity_block_len + 2 << " "
			<< (this->velocity_blocks.size() * velocity_block_len + 2)* 9 << std::endl;

		unsigned int j = 0;
		for (std::vector<unsigned int>::const_iterator
			block = this->velocity_block_list.begin();
			block != this->velocity_block_list.end();
			block++
		) {

			if (*block == error_velocity_block) {
				// assume no blocks after first error block
				break;
			}

			for (unsigned int z_index = 0; z_index < block_len_z; z_index++)
			for (unsigned int y_index = 0; y_index < block_len_y; y_index++)
			for (unsigned int x_index = 0; x_index < block_len_x; x_index++) {

				outfile << "8 ";
				for (int i = 0; i < 8; i++) {
					 outfile << j * 8 + i << " ";
				}
				outfile << std::endl;

				j++;
			}
		}
		outfile << "8 ";
		for (unsigned int i = 0; i < 8; i++) {
			outfile << j * 8 + i << " ";
		}
		outfile << std::endl;
		outfile << "8 ";
		for (unsigned int i = 0; i < 8; i++) {
			outfile << j * 8 + i << " ";
		}
		outfile << std::endl;

		// cell types
		outfile << "CELL_TYPES " << this->velocity_blocks.size() * velocity_block_len + 2 << std::endl;
		for (unsigned int i = 0; i < this->velocity_blocks.size() * velocity_block_len + 2; i++) {
			outfile << 11 << std::endl;
		}

		// Put minimum value from existing blocks into two additional cells
		double min_value = std::numeric_limits<double>::max();

		// distribution function
		outfile << "CELL_DATA " << this->velocity_blocks.size() * velocity_block_len + 2 << std::endl;
		outfile << "SCALARS rho double 1" << std::endl;
		outfile << "LOOKUP_TABLE default" << std::endl;
		for (std::vector<unsigned int>::const_iterator
			block = this->velocity_block_list.begin();
			block != this->velocity_block_list.end();
			block++
		) {

			if (*block == error_velocity_block) {
				// assume no blocks after first error block
				if (min_value == std::numeric_limits<double>::max()) {
					min_value = 0;
				}
				break;
			}

			const Velocity_Block* block_ptr = &(this->velocity_blocks.at(*block));

			for (unsigned int z_index = 0; z_index < block_len_z; z_index++)
			for (unsigned int y_index = 0; y_index < block_len_y; y_index++)
			for (unsigned int x_index = 0; x_index < block_len_x; x_index++) {

				const velocity_cell_indices_t indices = {x_index, y_index, z_index};
				const unsigned int velocity_cell = get_velocity_cell(indices);
				const double value = block_ptr->data[velocity_cell];
				outfile << value << " ";
				min_value = std::min(min_value, value);
			}
			outfile << std::endl;
		}
		outfile << min_value << " " << min_value << std::endl;

		if (!outfile.good()) {
			std::cerr << "Writing of vtk file probably failed" << std::endl;
			// TODO: throw an exception instead
			abort();
		}

		outfile.close();
	}


	/*!
	Adds an empty velocity block into this spatial cell.

	Returns true if given block was added or already exists.
	Returns false if given block is invalid or would be outside
	of the velocity grid.
	*/
	bool add_velocity_block(const unsigned int block) {
		if (block == error_velocity_block) {
			return false;
		}

		if (block >= max_velocity_blocks) {
			return false;
		}

		#ifndef NO_SPARSE
		if (this->velocity_blocks.count(block) > 0) {
			return true;
		}
		#endif

		#ifdef NO_SPARSE
		// assume blocks were added in default constructor
		#else
		this->velocity_blocks[block];
		this->block_address_cache[block] = &(this->velocity_blocks.at(block));
		#endif

		#ifdef NO_SPARSE
		Velocity_Block* block_ptr = &(this->velocity_blocks.at(block));
		#else
		Velocity_Block* block_ptr = this->block_address_cache[block];
		#endif

		block_ptr->clear();

		// set block parameters
		block_ptr->parameters[BlockParams::VXCRD] = get_velocity_block_vx_min(block);
		block_ptr->parameters[BlockParams::VYCRD] = get_velocity_block_vy_min(block);
		block_ptr->parameters[BlockParams::VZCRD] = get_velocity_block_vz_min(block);
		block_ptr->parameters[BlockParams::DVX] =
			(get_velocity_block_vx_max(block) - get_velocity_block_vx_min(block)) / 2;
		block_ptr->parameters[BlockParams::DVY] =
			(get_velocity_block_vy_max(block) - get_velocity_block_vy_min(block)) / 2;
		block_ptr->parameters[BlockParams::DVZ] =
			(get_velocity_block_vz_max(block) - get_velocity_block_vz_min(block)) / 2;

		// set neighbour pointers
		unsigned int neighbour_block;

		// -x direction
		neighbour_block = get_velocity_block(block, neg_x_dir);
		if (neighbour_block == error_velocity_block) {
			block_ptr->neighbors[neg_x_dir] = NULL;
		#ifndef NO_SPARSE
		} else if (this->velocity_blocks.count(neighbour_block) == 0) {
			block_ptr->neighbors[neg_x_dir] = &(this->null_block);
		#endif
		} else {
			block_ptr->neighbors[neg_x_dir] = &(this->velocity_blocks.at(neighbour_block));

			// update the neighbour list of neighboring block
			Velocity_Block* neighbour_ptr = &(this->velocity_blocks.at(neighbour_block));
			neighbour_ptr->neighbors[pos_x_dir] = block_ptr;
		}

		// +x direction
		neighbour_block = get_velocity_block(block, pos_x_dir);
		if (neighbour_block == error_velocity_block) {
			block_ptr->neighbors[pos_x_dir] = NULL;
		#ifndef NO_SPARSE
		} else if (this->velocity_blocks.count(neighbour_block) == 0) {
			block_ptr->neighbors[pos_x_dir] = &(this->null_block);
		#endif
		} else {
			block_ptr->neighbors[pos_x_dir] = &(this->velocity_blocks.at(neighbour_block));

			Velocity_Block* neighbour_ptr = &(this->velocity_blocks.at(neighbour_block));
			neighbour_ptr->neighbors[neg_x_dir] = block_ptr;
		}

		// -y direction
		neighbour_block = get_velocity_block(block, neg_y_dir);
		if (neighbour_block == error_velocity_block) {
			block_ptr->neighbors[neg_y_dir] = NULL;
		#ifndef NO_SPARSE
		} else if (this->velocity_blocks.count(neighbour_block) == 0) {
			block_ptr->neighbors[neg_y_dir] = &(this->null_block);
		#endif
		} else {
			block_ptr->neighbors[neg_y_dir] = &(this->velocity_blocks.at(neighbour_block));

			Velocity_Block* neighbour_ptr = &(this->velocity_blocks.at(neighbour_block));
			neighbour_ptr->neighbors[pos_y_dir] = block_ptr;
		}

		// +y direction
		neighbour_block = get_velocity_block(block, pos_y_dir);
		if (neighbour_block == error_velocity_block) {
			block_ptr->neighbors[pos_y_dir] = NULL;
		#ifndef NO_SPARSE
		} else if (this->velocity_blocks.count(neighbour_block) == 0) {
			block_ptr->neighbors[pos_y_dir] = &(this->null_block);
		#endif
		} else {
			block_ptr->neighbors[pos_y_dir] = &(this->velocity_blocks.at(neighbour_block));

			Velocity_Block* neighbour_ptr = &(this->velocity_blocks.at(neighbour_block));
			neighbour_ptr->neighbors[neg_y_dir] = block_ptr;
		}

		// -z direction
		neighbour_block = get_velocity_block(block, neg_z_dir);
		if (neighbour_block == error_velocity_block) {
			block_ptr->neighbors[neg_z_dir] = NULL;
		#ifndef NO_SPARSE
		} else if (this->velocity_blocks.count(neighbour_block) == 0) {
			block_ptr->neighbors[neg_z_dir] = &(this->null_block);
		#endif
		} else {
			block_ptr->neighbors[neg_z_dir] = &(this->velocity_blocks.at(neighbour_block));

			Velocity_Block* neighbour_ptr = &(this->velocity_blocks.at(neighbour_block));
			neighbour_ptr->neighbors[pos_z_dir] = block_ptr;
		}

		// +z direction
		neighbour_block = get_velocity_block(block, pos_z_dir);
		if (neighbour_block == error_velocity_block) {
			block_ptr->neighbors[pos_z_dir] = NULL;
		#ifndef NO_SPARSE
		} else if (this->velocity_blocks.count(neighbour_block) == 0) {
			block_ptr->neighbors[pos_z_dir] = &(this->null_block);
		#endif
		} else {
			block_ptr->neighbors[pos_z_dir] = &(this->velocity_blocks.at(neighbour_block));

			Velocity_Block* neighbour_ptr = &(this->velocity_blocks.at(neighbour_block));
			neighbour_ptr->neighbors[neg_z_dir] = block_ptr;
		}

		#ifndef NO_SPARSE
		unsigned int first_error_block = 0;
		while (first_error_block < max_velocity_blocks
		&& this->velocity_block_list[first_error_block] != error_velocity_block
		// block is in the list when preparing to receive blocks
		&& this->velocity_block_list[first_error_block] != block
		) {
			first_error_block++;
		}

		this->velocity_block_list[first_error_block] = block;
		#endif

		return true;
	}

	/*!
	Adds all velocity blocks than don't exist into the velocity grid.

	Returns true if all non-existing blocks were added, false otherwise.
	*/
	bool add_all_velocity_blocks(void)
	{
		bool result = true;

		for (unsigned int i = 0; i < max_velocity_blocks; i++) {
			#ifndef NO_SPARSE
			if (this->velocity_blocks.count(i) > 0) {
				continue;
			}
			#endif

			if (!this->add_velocity_block(i)) {
				result = false;
			}
		}

		return result;
	}


	/*!
	Removes given block from the velocity grid.
	Does nothing if given block doesn't exist.
	*/
	void remove_velocity_block(
		#ifdef NO_SPARSE
		const unsigned int /*block*/
		#else
		const unsigned int block
		#endif
	) {
		#ifndef NO_SPARSE
		if (block == error_velocity_block) {
			return;
		}

		if (this->velocity_blocks.count(block) == 0) {
			return;
		}

		// set neighbour pointers to this block to NULL
		unsigned int neighbour_block;

		// -x direction
		// TODO: use saved neighbor pointers instead of counting in the map
		neighbour_block = get_velocity_block(block, neg_x_dir);
		if (neighbour_block != error_velocity_block
		&& this->velocity_blocks.count(neighbour_block) > 0) {
			// TODO use cached addresses of neighbors
			Velocity_Block* neighbour_data = &(this->velocity_blocks.at(neighbour_block));
			neighbour_data->neighbors[pos_x_dir] = &(this->null_block);
		}

		// +x direction
		neighbour_block = get_velocity_block(block, pos_x_dir);
		if (neighbour_block != error_velocity_block
		&& this->velocity_blocks.count(neighbour_block) > 0) {
			Velocity_Block* neighbour_data = &(this->velocity_blocks.at(neighbour_block));
			neighbour_data->neighbors[neg_x_dir] = &(this->null_block);
		}

		// -y direction
		neighbour_block = get_velocity_block(block, neg_y_dir);
		if (neighbour_block != error_velocity_block
		&& this->velocity_blocks.count(neighbour_block) > 0) {
			Velocity_Block* neighbour_data = &(this->velocity_blocks.at(neighbour_block));
			neighbour_data->neighbors[pos_y_dir] = &(this->null_block);
		}

		// +y direction
		neighbour_block = get_velocity_block(block, pos_y_dir);
		if (neighbour_block != error_velocity_block
		&& this->velocity_blocks.count(neighbour_block) > 0) {
			Velocity_Block* neighbour_data = &(this->velocity_blocks.at(neighbour_block));
			neighbour_data->neighbors[neg_y_dir] = &(this->null_block);
		}

		// -z direction
		neighbour_block = get_velocity_block(block, neg_z_dir);
		if (neighbour_block != error_velocity_block
		&& this->velocity_blocks.count(neighbour_block) > 0) {
			Velocity_Block* neighbour_data = &(this->velocity_blocks.at(neighbour_block));
			neighbour_data->neighbors[pos_z_dir] = &(this->null_block);
		}

		// +z direction
		neighbour_block = get_velocity_block(block, pos_z_dir);
		if (neighbour_block != error_velocity_block
		&& this->velocity_blocks.count(neighbour_block) > 0) {
			Velocity_Block* neighbour_data = &(this->velocity_blocks.at(neighbour_block));
			neighbour_data->neighbors[neg_z_dir] = &(this->null_block);
		}

		this->velocity_blocks.erase(block);
		this->block_address_cache[block] = &(this->null_block);

		/*
		Move the last existing block in the block list
		to the removed block's position
		*/
		unsigned int block_index = 0;
		while (block_index < max_velocity_blocks
		&& this->velocity_block_list[block_index] != block) {
			block_index++;
		}
		//debug
		if (block_index == max_velocity_blocks) {
			std::cerr << "Velocity block " << block << " not found in list" << std::endl;
			abort();
		}

		unsigned int first_error_block = 0;
		while (first_error_block < max_velocity_blocks
		&& this->velocity_block_list[first_error_block] != error_velocity_block) {
			first_error_block++;
		}

		if (block_index == first_error_block - 1) {
			this->velocity_block_list[block_index] = error_velocity_block;
		} else {
			this->velocity_block_list[block_index] = this->velocity_block_list[first_error_block - 1];
			this->velocity_block_list[first_error_block - 1] = error_velocity_block;
		}
		#endif
	}


	/*!
	Removes all velocity blocks from this spatial cell.
	*/
	void clear(void)
	{
		#ifndef NO_SPARSE
		this->velocity_blocks.clear();
		for (unsigned int i = 0; i < max_velocity_blocks; i++) {
			this->block_address_cache[i] = &(this->null_block);
			this->velocity_block_list[i] = error_velocity_block;
		}
		#endif
	}


	/*!
	Prepares this spatial cell to receive the velocity grid over MPI.
	*/
	void prepare_to_receive_blocks(void)
	{
		#ifndef NO_SPARSE
		unsigned int number_of_blocks = 0;
		while (number_of_blocks < max_velocity_blocks
		&& this->velocity_block_list[number_of_blocks] != error_velocity_block) {
			number_of_blocks++;
		}

		// add_velocity_block overwrites the block list every time so:
		std::vector<unsigned int> old_block_list(number_of_blocks, error_velocity_block);
		for (unsigned int block_index = 0; block_index < number_of_blocks; block_index++) {
			old_block_list[block_index] = this->velocity_block_list[block_index];
		}

		this->clear();

		// add velocity blocks that are about to be received with MPI
		for (unsigned int block_index = 0; block_index < number_of_blocks; block_index++) {
			this->add_velocity_block(old_block_list[block_index]);
		}
		#endif
	}


	/*!
	Sets the type of data to transfer by mpi_datatype.
	*/
	static void set_mpi_transfer_type(const int type)
	{
		Spatial_Cell::mpi_transfer_type = type;
	}

	/*!
	Gets the type of data that will be transferred by mpi_datatype.
	*/
	static int get_mpi_transfer_type(void)
	{
		return Spatial_Cell::mpi_transfer_type;
	}



private:

	/*
	Which data is transferred by the mpi datatype given by spatial cells.

	If 0 no data is transferred
	If 1 velocity block lists are transferred
	If 2 velocity block data is transferred
	*/
	static int mpi_transfer_type;

	/*
	Minimum value of distribution function
	in any cell of a velocity block for the
	block to be considered to have contents
	*/
	double velocity_block_min_value;

	/*
	Minimum value of the average of distribution
	function within a velocity block for the
	block to be considered to have contents
	*/
	double velocity_block_min_avg_value;

	/*!
	Used as a neighbour instead of blocks that don't
	exist but would be inside of the velocity grid.
	Neighbors that would be outside of the grid are always NULL.
	*/
	Velocity_Block null_block;

	// data of velocity blocks that exist in this cell
	#ifdef NO_SPARSE
	std::vector<Velocity_Block> velocity_blocks;
	#else
	boost::unordered_map<unsigned int, Velocity_Block> velocity_blocks;
	/*
	Speed up search of velocity block addresses in the hash table above.
	Addresses of non-existing blocks point to the null block.
	*/
	std::vector<Velocity_Block*> block_address_cache;
	#endif



public:

	/*
	List of velocity blocks in this cell, used for
	transferring spatial cells between processes using MPI.
	*/
	std::vector<unsigned int> velocity_block_list;

	/*
	Bulk variables in this spatial cell.
	*/
	std::vector<double> parameters;

}; // class Spatial_Cell

int Spatial_Cell::mpi_transfer_type = 0;

}} // namespaces
#endif

