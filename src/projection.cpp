#include <cmath>
#include <iostream>

#include "projection.hpp"

namespace proj {

// - static values
GridRenderInfo renderInfo;

// - dynamic values
// default density is 1:1
uvec2 cellDensity(1, 1);
ulim2 gridLimX, gridLimY;

// - private members

// This modifies the param delta in order to limit it!
ulim2 translateLimits(float *delta, ulim2 reference, int hardLimit = 0);

void init() {
    controls::minScale = 1;

    // define sizes & proportions
    if (config::noDownsample) {
        renderInfo.numVertices = {config::cols, config::rows};
        renderInfo.totalVertices = config::cols * config::rows;
        renderInfo.cellDensity = {1, 1};

        controls::maxScale = 100.0f;
        controls::scaleFactor = 0.99f; // related to render space

        controls::translateFactor = 0.002f; // related to render space
    } else {
        renderInfo.numVertices.x =
            config::cols > config::width ? config::width : config::cols;
        renderInfo.numVertices.y =
            config::rows > config::height ? config::height : config::rows;
        renderInfo.totalVertices =
            renderInfo.numVertices.x * renderInfo.numVertices.y;
        // max density
        renderInfo.cellDensity.x = config::cols / renderInfo.numVertices.x;
        renderInfo.cellDensity.y = config::rows / renderInfo.numVertices.y;

        // max scale will give us 1:1 cell to vertice mapping
        controls::maxScale =
            std::max(renderInfo.cellDensity.x, renderInfo.cellDensity.y);
        // scale factor should give us density steps of 1
        controls::scaleFactor =
            std::min(renderInfo.numVertices.x, renderInfo.numVertices.y);

        controls::translateFactor = 10.0f; // related to cell space
    }

    // define default gridEnd (which is defined at run-time)
    gridLimX = {0, config::cols};
    gridLimY = {0, config::rows};
}

void update() {
    // if downsampling is disabled, there's nothing to do
    if (config::noDownsample)
        return;

    // section size will start with the whole grid (scale = 1)
    uvec2 sectionSize((uint)std::ceil(config::cols / float(controls::scale)),
                      (uint)std::ceil(config::rows / float(controls::scale)));
    // how many grid cells will be mapped to each vertice
    cellDensity = {
        (uint)std::floor(sectionSize.x / float(renderInfo.numVertices.x)),
        (uint)std::floor(sectionSize.y / float(renderInfo.numVertices.y))};
    // the indices of the considered grid sections
    const ulim2 refX((config::cols / 2.0) - sectionSize.x / 2,
                     (config::cols / 2.0) + sectionSize.x / 2);
    const ulim2 refY((config::rows / 2.0) - sectionSize.y / 2,
                     (config::rows / 2.0) + sectionSize.y / 2);

    // calculate translations
    gridLimX = translateLimits(&controls::position.x, refX, config::cols);
    gridLimY = translateLimits(&controls::position.y, refY, config::rows);

    std::cout << std::endl
              << "sec xy " << sectionSize.x << "," << sectionSize.y     //
              << " / dens xy " << cellDensity.x << "," << cellDensity.y //
              << " / grid x " << gridLimX.start << "-" << gridLimX.end  //
              << " / grid y " << gridLimY.start << "-" << gridLimY.end  //
              << " / maxX " << (gridLimX.range()) / cellDensity.x       //
              << " / maxY " << (gridLimY.range()) / cellDensity.y       //
              << std::endl;
}

uint getVerticeIdx(uvec2 gridPos) {
    if (config::noDownsample)
        // no conversion needed, grid index = vertice index
        return gridPos.y * config::cols + gridPos.x;

    uint vx = (gridPos.x - gridLimX.start) / cellDensity.x;
    uint vy = (gridPos.y - gridLimY.start) / cellDensity.y;
    // return a position when the mapping is valid
    if (vx < renderInfo.numVertices.x && vy < renderInfo.numVertices.y)
        return vy * renderInfo.numVertices.x + vx;
    // otherwise return a default position
    return 0;
}

// This modifies the param delta in order to limit it!
ulim2 translateLimits(float *delta, ulim2 reference, int hardLimit) {
    // int, since it can be negative
    int translated;

    if (*delta > 0) {
        // make the end of the grid invisible by shifting the beginning
        // of the vector as indicated by delta
        translated = reference.end + *delta;
        // if delta is positive we may have exploded up
        if (translated > hardLimit) {
            // return to the limit
            translated = hardLimit;
            // and limit the knob
            *delta = (translated - int(reference.end));
        }
        // only modify the begin limit by the amount modified on the ending
        return {reference.start + (translated - reference.end),
                (uint)translated};
    } else {
        // make the start of the grid visible by considering the beginning
        // of the vector as indicated by delta (which is negative)
        translated = reference.start + *delta;
        // if delta is negative we may have exploded down
        if (translated < 0) {
            // return to the limit
            translated = 0;
            // and limit the knob
            *delta = (translated - int(reference.start));
        }
        // only modify the end limit by the amount modified on the beginning
        return {(uint)translated,
                reference.end - (translated - reference.start)};
    }
}

} // namespace proj