#include <stdio.h>
#include <stdlib.h>

#ifndef data_h
#define data_h


typedef struct DataWrite {
    FILE *fpt;
    const char *path;
} datawrite;


void openCSV(datawrite *dw);

void closeCSV(datawrite *dw);

void writeCSVHeader(datawrite *dw);

typedef struct CSVEntry {
    int id;
    int n;
    const char *kernel_name;
    double exec_time;
} csventry;

void writeCSVEntry(datawrite *dw, csventry *entry);

#endif