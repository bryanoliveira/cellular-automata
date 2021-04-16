#ifndef GRID_CPU_HPP_
#define GRID_CPU_HPP_

#include "grid_interface.hpp"

namespace cpu {

template <typename GridType> class Grid : public GridInterface<Grid, GridType> {
  protected:
    bool mUseGpu;
    size_t mSize;
    GridType *mData;

  public:
    Grid(size_t pSize, bool init, GridType initValue = false);
    ~Grid();
    void swap(Grid &pGrid);
    GridType get(unsigned int position);

    GridType &operator[](unsigned int position) const {
        return mData[position];
    }
};

} // namespace cpu

#endif