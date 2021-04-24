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

void init() {
    // define sizes & proportions
    if (config::noDownsample) {
        renderInfo.numVertices = {config::cols, config::rows};
        renderInfo.totalVertices = config::cols * config::rows;
        renderInfo.cellDensity = {1, 1};
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
    }

    // define default gridEnd (which is defined at run-time)
    gridEnd.x = config::cols;
    gridEnd.y = config::rows;

    // config controls mins and maxes
    controls::minScale = 1;
    controls::maxScale =
        std::max(renderInfo.cellDensity.x, renderInfo.cellDensity.y);
}

void update() {
    // TODO check unsigned/signed comparisons and value usage

    // if downsampling is disabled, there's nothing to do
    if (config::noDownsample)
        return;

    // section size will start with the whole grid (scale = 1)
    int sectionSizeX = std::ceil(config::cols / float(controls::scale));
    int sectionSizeY = std::ceil(config::rows / float(controls::scale));
    // how many grid cells will be mapped to each vertice
    cellDensity.x = std::floor(sectionSizeX / float(renderInfo.numVertices.x));
    cellDensity.y = std::floor(sectionSizeY / float(renderInfo.numVertices.y));
    // the indices of the considered grid sections
    int startX = (config::cols / 2.0) - sectionSizeX / 2;
    int endX = (config::cols / 2.0) + sectionSizeX / 2;
    int startY = (config::rows / 2.0) - sectionSizeY / 2;
    int endY = (config::rows / 2.0) + sectionSizeY / 2;

    // calculate x translation
    if (controls::position[0] < 0) {
        // make the start of the grid visible by considering the beginning
        // of the vector as indicated by the position - note: position is
        // negative
        gridStart.x = startX + controls::position[0];
        if (gridStart.x < 0) {
            gridStart.x = 0;
            // limit the knob
            controls::position[0] = (gridStart.x - startX);
        }
        // only modify the end limit by the amount modified on the beginning
        // note: this will not be negative
        gridEnd.x = endX - (gridStart.x - startX);
    } else {
        // if position is positive the grid will be at the left of the
        // canvas so we extend the considered end of the vector
        gridEnd.x = endX + controls::position[0];
        if (gridEnd.x > (int)config::cols) {
            gridEnd.x = config::cols;
            // limit the knob
            controls::position[0] = (gridEnd.x - endX);
        }
        // and crop the start position by the amount modified on the end
        gridStart.x = startX + (gridEnd.x - endX);
    }

    // calculate y translation
    if (controls::position[1] < 0) {
        // make the start of the grid visible by considering the beginning
        // of the vector as indicated by the position - note: position is
        // negative
        gridStart.y = startY + controls::position[1];
        if (gridStart.y < 0) {
            gridStart.y = 0;
            // limit the knob
            controls::position[1] = gridStart.y - startY;
        }
        // only modify the end limit by the amount modified on the beginning
        // note: this will not be negative
        gridEnd.y = endY - (gridStart.y - startY);
    } else {
        // if position is positive the grid will be at the left of the
        // canvas so we extend the considered end of the vector
        gridEnd.y = endY + controls::position[1];
        if (gridEnd.y > (int)config::rows) {
            gridEnd.y = config::rows;
            // limit the knob
            controls::position[1] = gridEnd.y - endY;
        }
        // and crop the start position by the amount modified on the end
        gridStart.y = startY + (gridEnd.y - endY);
    }

    std::cout << std::endl
              << "pos xy " << controls::position[0] << ","
              << controls::position[1]                               //
              << " / sec xy " << sectionSizeX << "," << sectionSizeY //
              << " / grid x " << gridStart.x << "-" << gridEnd.x     //
              << " / grid y " << gridStart.y << "-"
              << gridEnd.y //
              // << " / vert x " << vStartX << "-" << vEndX //
              // << " / vert y " << vStartY << "-" << vEndY          //
              // << " / dens xy " << densityX << "," << densityY //
              //   << " / map xy" << sectionSizeX / densityX << ", "
              //   << sectionSizeY / densityY //
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

} // namespace proj