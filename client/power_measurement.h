#ifndef POWER_MEASUREMENT_HEADER
#define POWER_MEASUREMENT_HEADER

#include <stdlib.h>
#include <stdint.h>
#include <dirent.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <sys/wait.h>
#include <time.h>

#define GET_NS(t) t.tv_sec * 1000000000 + t.tv_nsec

//These are specific to ZCU102
#define VCCPSINTFP 0
#define VCCINTLP 1
#define VCCPSAUX 2
#define VCCPSPLL 3
#define MGTRAVCC 4
#define MGTRAVTT 5
#define VCCPSDDR 6
#define VCCOPS 7
#define VCCOPS3 8
#define VCCPSDDRPLL 9
#define VCCINT  10
#define VCCBRAM 11
#define VCCAUX 12
#define VCC1V2 13
#define VCC3V3 14
#define VADJ_FMC 15
#define MGTAVCC 16
#define MGTAVTT 17

const char railname_arr[50][12] = { "VCCPSINTFP", "VCCINTLP", "VCCPSAUX",
		"VCCPSPLL", "MGTRAVCC", "MGTRAVTT", "VCCPSDDR", "VCCOPS", "VCCOPS3",
		"VCCPSDDRPLL", "VCCINT", "VCCBRAM", "VCCAUX", "VCC1V2", "VCC3V3",
		"VADJ_FMC", "MGTAVCC", "MGTAVTT" };

typedef struct ina {

	char current_path[50];
	char voltage_path[50];
	char power_path[50];
	char name[12];
	int current;
	int voltage;
	int power;
	int last;

} ina;

int cmp_ina(const void *a, const void *b);

void populate_ina_array(ina *inas);

void list_inas(ina *inas);

void run_bm(int verbose, ina *inas);

#endif
