#include <algorithm>
#include "grid_cpu.hpp"

namespace cpu {

Grid::Grid(size_t pSize, bool init, GridType initValue) {
    mSize = pSize * sizeof(GridType);

    mData = (GridType *)malloc(mSize);
    if (init)
        std::fill(&mData[0], &mData[pSize - 1], initValue);
}

Grid::~Grid() { free(mData); }

void Grid::swap(Grid &pGrid) {
    GridType *tempData = mData;
    mData = pGrid.mData;
    pGrid.mData = tempData;
}

GridType Grid::get(unsigned int position) { return mData[position]; }

} // namespace cpu