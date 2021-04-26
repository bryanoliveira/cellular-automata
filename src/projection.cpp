#include <cmath>
#include <iostream>

#include "projection.hpp"

namespace proj {

// - static values
GridRenderInfo info;

// - dynamic values
// default density is 1:1
uvec2 cellDensity(1, 1);
ulim2 gridLimX, gridLimY;

// - private members

// This modifies the param delta in order to limit it!
ulim2 translateLimits(float *delta, ulim2 ref, int sectionSize, int hardLimit);

void init() {
    controls::minScale = 1;

    // define sizes & proportions
    if (config::noDownsample) {
        info.numVertices = {config::cols, config::rows};
        info.totalVertices = config::cols * config::rows;
        info.cellDensity = {1, 1};

        controls::maxScale = 100.0f;
        controls::scaleFactor = 0.99f; // related to render space

        controls::translateFactor = 0.002f; // related to render space
    } else {
        info.numVertices.x =
            config::cols > config::width ? config::width : config::cols;
        info.numVertices.y =
            config::rows > config::height ? config::height : config::rows;
        info.totalVertices = info.numVertices.x * info.numVertices.y;
        // max density
        info.cellDensity.x = config::cols / info.numVertices.x;
        info.cellDensity.y = config::rows / info.numVertices.y;

        // max scale will give us 1:1 cell to vertice mapping
        controls::maxScale = std::max(info.cellDensity.x, info.cellDensity.y);
        // scale factor should give us density steps of 1
        controls::scaleFactor = 1;

        controls::translateFactor =
            config::cols / float(info.numVertices.x); // related to cell space
    }

    // define default gridEnd (which is defined at run-time)
    gridLimX = {0, config::cols};
    gridLimY = {0, config::rows};
}

void update() {
    // if downsampling is disabled, there's nothing to do
    if (config::noDownsample)
        return;

    // how many grid cells will be mapped to each vertice
    cellDensity = {
        uint(std::max(config::cols / info.numVertices.x - controls::scale,
                      1.0f)),
        uint(std::max(config::rows / info.numVertices.y - controls::scale,
                      1.0f))};

    // section size will start with the whole grid (scale = 1)
    uvec2 sectionSize(info.numVertices.x * cellDensity.x,
                      info.numVertices.y * cellDensity.y);
    // the indices of the considered grid sections
    const ulim2 refX((config::cols / 2.0) - sectionSize.x / 2,
                     (config::cols / 2.0) + sectionSize.x / 2);
    const ulim2 refY((config::rows / 2.0) - sectionSize.y / 2,
                     (config::rows / 2.0) + sectionSize.y / 2);

    // calculate translations
    gridLimX = translateLimits(&controls::position.x, refX, sectionSize.x,
                               config::cols);
    gridLimY = translateLimits(&controls::position.y, refY, sectionSize.y,
                               config::rows);

    // std::cout << std::endl
    //           << "sec xy " << sectionSize.x << "," << sectionSize.y     //
    //           << " / dens xy " << cellDensity.x << "," << cellDensity.y //
    //           << " / grid x " << gridLimX.start << "-" << gridLimX.end  //
    //           << " / grid y " << gridLimY.start << "-" << gridLimY.end  //
    //           << " / maxX " << (gridLimX.range()) / cellDensity.x       //
    //           << " / maxY " << (gridLimY.range()) / cellDensity.y       //
    //           << std::endl;
}

uint getVerticeIdx(uvec2 gridPos) {
    if (config::noDownsample)
        // no conversion needed, grid index = vertice index
        return gridPos.y * config::cols + gridPos.x;

    uint vx = (gridPos.x - gridLimX.start) / cellDensity.x;
    uint vy = (gridPos.y - gridLimY.start) / cellDensity.y;
    // return a position when the mapping is valid
    if (vx < info.numVertices.x && vy < info.numVertices.y)
        return vy * info.numVertices.x + vx;
    // std::cout << "out of bounds" << std::endl;
    // otherwise return a default position
    return 0;
}

// This modifies the param delta in order to limit it!
ulim2 translateLimits(float *delta, ulim2 ref, int sectionSize, int hardLimit) {
    ulim2 translated = {
        uint(std::min(std::max(int(ref.start + *delta), 0), hardLimit)),
        uint(std::max(std::min(int(ref.end + *delta), hardLimit), sectionSize)),
    };

    // crop delta by the amount used
    if (*delta < 0)
        // this may be negative - convert it before the operation
        *delta = int(translated.start) - int(ref.start);
    else if (*delta > 0)
        // this will be always positive
        *delta = translated.end - ref.end;

    return translated;
}

} // namespace proj