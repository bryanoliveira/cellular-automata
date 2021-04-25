#include <cmath>
#include <iostream>

#include "projection.hpp"

namespace proj {

// - static values
GridRenderInfo renderInfo;

// - dynamic values
// default density is 1:1
uvec2 cellDensity(1, 1);
// we will start to render from the first vertices
uvec2 vStart(0, 0);
// we will start to render from the first grid cell
uvec2 gridStart(0, 0);
// until the last grid cell (set in init)
uvec2 gridEnd;

// - private members

// This modifies the param delta in order to limit it!
ulim2 translateLimits(float *delta, uint reference1, uint reference2,
                      int hardLimit = 0);

void init() {
    controls::minScale = 1;

    // define sizes & proportions
    if (config::noDownsample) {
        renderInfo.numVertices = {config::cols, config::rows};
        renderInfo.totalVertices = config::cols * config::rows;
        renderInfo.cellDensity = {1, 1};

        controls::maxScale = 100.0f;
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

        controls::maxScale =
            std::max(renderInfo.cellDensity.x, renderInfo.cellDensity.y);
        controls::translateFactor = 10.0f; // related to cell space
    }

    // define default gridEnd (which is defined at run-time)
    gridEnd.x = config::cols;
    gridEnd.y = config::rows;
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
    const uvec2 refStart((config::cols / 2.0) - sectionSize.x / 2,
                         (config::rows / 2.0) - sectionSize.y / 2);
    const uvec2 refEnd((config::cols / 2.0) + sectionSize.x / 2,
                       (config::rows / 2.0) + sectionSize.y / 2);

    ulim2 gridLim;
    // calculate x translation
    if (controls::position.x < 0)
        gridLim = translateLimits(&controls::position.x, refStart.x, refEnd.x);
    else
        gridLim = translateLimits(&controls::position.x, refEnd.x, refStart.x,
                                  config::cols);
    gridStart.x = gridLim.start;
    gridEnd.x = gridLim.end;
    // calculate y translation
    if (controls::position.y < 0)
        gridLim = translateLimits(&controls::position.y, refStart.y, refEnd.y);
    else
        gridLim = translateLimits(&controls::position.y, refEnd.y, refStart.y,
                                  config::rows);
    gridStart.y = gridLim.start;
    gridEnd.y = gridLim.end;

    std::cout << std::endl
              << "pos xy " << controls::position.x << ","
              << controls::position.y                                  //
              << " / sec xy " << sectionSize.x << "," << sectionSize.y //
              << " / grid x " << gridStart.x << "-" << gridEnd.x       //
              << " / grid y " << gridStart.y << "-"
              << gridEnd.y //
              // << " / vert x " << vStartX << "-" << vEndX //
              // << " / vert y " << vStartY << "-" << vEndY          //
              // << " / dens xy " << densityX << "," << densityY //
              //   << " / map xy" << sectionSize.x / densityX << ", "
              //   << sectionSize.y / densityY //
              << " / maxX "
              << (gridEnd.x - 1 - gridStart.x) / cellDensity.x + vStart.x
              // << " / cmaxV "
              //   << ((endY - startY - 1) / densityY) *
              //   mRenderInfo.numVerticesX +
              //          (endX - 1) / densityX
              // << " - maxV " << mRenderInfo.numVertices
              << std::endl;
}

uint getVerticeIdx(uvec2 gridPos) {
    if (config::noDownsample)
        // no conversion needed, grid index = vertice index
        return gridPos.y * config::cols + gridPos.x;

    uint vx = (gridPos.x - gridStart.x) / cellDensity.x + vStart.x;
    uint vy = (gridPos.y - gridStart.y) / cellDensity.y + vStart.y;
    // return a position when the mapping is valid
    if (vx < renderInfo.numVertices.x && vy < renderInfo.numVertices.y)
        return vy * renderInfo.numVertices.x + vx;
    // otherwise return a default position
    return 0;
}

// This modifies the param delta in order to limit it!
ulim2 translateLimits(float *delta, uint reference1, uint reference2,
                      int hardLimit) {
    // make the start of the grid visible by considering the beginning
    // of the vector as indicated by the position - note: delta may be
    // negative
    // TODO avoid ifs using math
    int translated = reference1 + *delta;
    if (delta < 0) {
        // if delta is negative we may have exploded down
        if (translated < hardLimit)
            translated = hardLimit;
    } else {
        // if delta is positive we may have exploded up
        if (translated > hardLimit)
            translated = hardLimit;
    }
    // limit the knob
    *delta = (translated - reference1);
    uint start = translated;
    // only modify the end limit by the amount modified on the beginning
    uint end = reference2 - (start - reference1);
    return {start, end};
}

} // namespace proj