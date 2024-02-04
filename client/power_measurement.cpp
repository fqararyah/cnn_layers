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

int cmp_ina(const void *a, const void *b) {
	ina *temp1 = (ina*) a;
	ina *temp2 = (ina*) b;
	int len1 = strlen(temp1->current_path);
	int len2 = strlen(temp2->current_path);

	if (len1 == len2) {
		return strcmp(temp1->current_path, temp2->current_path);
	} else if (len1 > len2) {
		return 1;
	} else {
		return -1;
	}

}

void populate_ina_array(ina *inas) {
	DIR *d;
	struct dirent *dir;

	char buffer[100];
	char fname_buff[100];

	FILE *fptr;

	d = opendir("/sys/class/hwmon/");
	int counter = 0;

	while ((dir = readdir(d)) != NULL) {
		if (strncmp(".", dir->d_name, 1) == 0) {
			continue;
		}
		//printf("tree: %s\n", dir->d_name);
		strcpy(fname_buff, "/sys/class/hwmon/");
		strcat(fname_buff, dir->d_name);
		strcat(fname_buff, "/name");

		//printf("name: %s\n", fname_buff);

		fptr = fopen(fname_buff, "r");
		fread(&buffer, 10, 1, fptr);
		//printf("device type: %s", buffer);

		if (strncmp(buffer, "ina", 3) == 0) {
			fname_buff[strlen(fname_buff) - 5] = 0;

			strcpy(inas[counter].current_path, fname_buff);
			strcat(inas[counter].current_path, "/curr1_input");

			strcpy(inas[counter].voltage_path, fname_buff);
			strcat(inas[counter].voltage_path, "/in1_input");

			strcpy(inas[counter].power_path, fname_buff);
			strcat(inas[counter].power_path, "/power1_input");

//			printf("found: %s\n", inas[counter].ina_dir);
			inas[counter].last = 0;
			counter++;
		}

	}

	qsort(inas, counter, sizeof(ina), cmp_ina);
	if (counter > 0)
		inas[counter - 1].last = 1;

	counter = 0;
	while (1) {
		sprintf(inas[counter].name, railname_arr[counter]);
		if (inas[counter].last == 1)
			return;

		counter++;
	}

	closedir(d);

}

void list_inas(ina *inas) {
	int counter = 0;
	while (1) {
		printf("Found INA%03d at dir: %s\n", counter,
				inas[counter].current_path);
		if (inas[counter].last == 1)
			break;

		counter++;
	}
	return;
}

void run_bm(float &plpower, float &pspower, float &mgtpower, ina *inas) {
	FILE *ina_ptr;

	struct timespec time_s;

	char buffer[20];

	int counter = 0;
	while (1) {

		ina_ptr = fopen(inas[counter].power_path, "r");

		fscanf(ina_ptr, "%[^\n]", buffer);

		inas[counter].power = atoi(buffer);

//		if (verbose == 1) {
//			printf("Power # %d = %d \n", counter, atoi(buffer));
//		}

		if (inas[counter].last) {

//			clock_gettime(CLOCK_REALTIME, &time_s);
//			printf("%ld,", GET_NS(time_s));

			pspower = (float) (inas[VCCPSINTFP].power + inas[VCCINTLP].power
					+ inas[VCCPSAUX].power + inas[VCCPSPLL].power
					+ inas[VCCPSDDR].power +
					//inas[VCCOPS].power+
					//inas[VCCOPS3].power+
					inas[VCCPSDDRPLL].power) / 1000000.0;

			printf(" %.3f,", pspower);

			plpower = (float) (inas[VCCINT].power + inas[VCCBRAM].power
					+ inas[VCCAUX].power + inas[VCC1V2].power
					+ inas[VCC3V3].power) / 1000000.0;

			printf(" %.3f,", plpower);

			mgtpower = (float) (inas[MGTRAVCC].power + inas[MGTRAVTT].power
					+ inas[MGTAVCC].power + inas[MGTAVTT].power
					+ inas[VCC3V3].power) / 1000000.0;

			printf(" %.3f,", mgtpower);

			printf(" %.3f\n", mgtpower + plpower + pspower);

			fclose(ina_ptr);
			break;
		}

		fclose(ina_ptr);

		counter++;

	}

}
