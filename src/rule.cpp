#include <iostream>
#include <fstream>
#include <sstream>

#define STATE_INIT 0
#define STATE_COMMENT 1
#define STATE_HEADER_X 2
#define STATE_HEADER_Y 3
#define STATE_RULE 4
#define STATE_GRID 5
#define STATE_END 6

void load_rule(unsigned int posX, unsigned int posY) {
    unsigned long x, y, row, col;

    std::ifstream infile("rules/glider.rle");
    std::stringstream buffer;
    char ch;

    unsigned short state = STATE_INIT;

    while (infile >> ch) {
        std::cout << "buf: " << ch << " | state: " << state << std::endl;

        switch (state) {
        case STATE_INIT:
            // line init can be a comment or the file header
            if (ch == '#')
                state = STATE_COMMENT;
            else if (ch == 'x')
                state = STATE_HEADER_X;
            break;
        case STATE_COMMENT:
            // ignore all chars, except for endl
            if (ch = '\n')
                state = STATE_INIT;
            break;
        case STATE_HEADER_X:
            // after we read the char "x" we skip the " = "
            if (ch == ' ' || ch == '=')
                continue;
            // if we reached ",", process the buffer into a number
            else if (ch = ',') {
                buffer >> x;
                buffer.clear();
                // the next number we'll read is Y
                state = STATE_HEADER_Y;
            }
            // read the number
            else
                buffer << ch;
            break;
        case STATE_HEADER_Y:
            // skip "y", "=" and " "
            if (ch == 'y' || ch == '=' || ch == ' ')
                continue;
            // if we reached ",", process the buffer into a number
            else if (ch = ',') {
                buffer >> y;
                buffer.clear();
                // the next thing is the rule
                state = STATE_RULE;
            }
            // read the number
            else
                buffer << ch;
            break;
        case STATE_RULE:
            // we don't read the rules yet
            if (ch == '\n')
                state = STATE_GRID;
            break;
        case STATE_GRID:
            // put numbers into buffer
            if (ch >= '0' && ch <= '9')
                buffer << ch;
            // parse the current buffer
            else if (ch == 'o' || ch == 'b') {
                bool alive = ch == 'o';
                // TODO parse the buffer
                col++;
            } else if (ch == '$')
                row++;
            break;
        }
    }

    std::cout << "x: " << x << " y: " << y << std::endl;
}