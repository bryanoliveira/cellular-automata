#ifndef GRID_INTERFACE_HPP_
#define GRID_INTERFACE_HPP_

template <class ImplClass, typename GridType> class GridInterface {
  public:
    virtual ~GridInterface();
    virtual void swap(ImplClass &pGrid) = 0;
    virtual GridType get(unsigned int position) = 0;

    virtual GridType &operator[](unsigned int position) const = 0;
};

extern GridInterface &grid;
extern GridInterface &nextGrid;

#endif