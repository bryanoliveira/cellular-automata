#include<stdio.h>


void printBits(unsigned char num)
{
   for(int bit=0;bit<(sizeof(unsigned char) * 8); bit++)
   {
      printf("%i ", num & 0x01);
      num = num >> 1;
   }
}

int main() {
    unsigned char byte = 5;
    printBits(byte);
    printf("\n\n");
    for (int bit = 0; bit < 8; ++bit) {
        unsigned char num = 1 << bit;
        printBits(num);
        printf("\n%d\n\n", (byte & (1 << bit)) != 0);
    }
    return 0;
}