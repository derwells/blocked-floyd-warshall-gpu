#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

// Custom library
#include "data.h"

#define CSV_HEADER "id,size,kernel,exec_time\n"
#define ROW_TEMPLATE "%d,%d,%s,%lf\n"


void openCSV(datawrite *dw) {
    dw->fpt = fopen(dw->path, "w+");
}

void closeCSV(datawrite *dw) {
    if (dw->fpt != NULL) 
        fclose(dw->fpt);
}

void writeCSVHeader(datawrite *dw) {
    fprintf(dw->fpt, CSV_HEADER);
}

void writeCSVEntry(datawrite *dw, csventry *entry) {
    fprintf(dw->fpt, ROW_TEMPLATE, entry->id, entry->n, entry->kernel_name, entry->exec_time);
}
