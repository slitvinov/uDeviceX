/*
 *  fitfun.c
 *  Pi4U
 *
 *  Created by Lina Kulakova on 16/9/15.
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <ftw.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <stdlib.h>
#include <torc.h>
#include "gsl_headers.h"
#include "spawner.h"

#define BUF_LEN 1024

static pthread_mutex_t fork_mutex = PTHREAD_MUTEX_INITIALIZER;

static const int display = 1;
static const int debug = 1;
static const double fail = -1e12;

static const int sim_n = 8;
static const int sim_d = 7;

#define S(i, j) (sim_data[(i) * sim_d + (j)])
#define E(i, j) (exp_data[(i) * exp_d + (j)])

void write_task_file(char *taskdir, double *par, int n) {
    int i;
    FILE *fp;
    char cwd[BUF_LEN];

    getcwd(cwd, sizeof(cwd));
    chdir(taskdir);
    fp = fopen("points.txt", "w");
    if (!fp) {
        printf("Cannot create file in %s\n", taskdir); fflush(0);
        abort();
    }
    fprintf(fp, "RBCgammaC _gamma_dot\n");
    for (i = 0; i < n; ++i) fprintf(fp, "%.16f ", par[i]); fprintf(fp, "\n");
    fclose(fp);
    chdir(cwd);
}

void pre_task(char *taskdir, double *par, int n) {
    char cwd[BUF_LEN], from_dir[BUF_LEN];

    getcwd(cwd, sizeof(cwd));
    chdir(taskdir);

    snprintf(from_dir, BUF_LEN, "%s/../../to_copy", taskdir);
    if (copy_from_dir(from_dir) != 0) {
        printf("Error in copy from dir %s\n", from_dir);
        abort();
    }
    chdir(cwd);
}

void run_task(char *taskdir, double *par, int n) {
    int rf, fd, status; 
    char line[BUF_LEN], *largv[2], cwd[BUF_LEN];

    getcwd(cwd, sizeof(cwd));
    chdir(taskdir);
    while (pthread_mutex_trylock(&fork_mutex) == EBUSY) usleep(1e6);
    rf = fork();
    if (rf < 0) {
        printf("Fork failed\n"); fflush(0);
    } else if (rf == 0) {
        sprintf(line, "./sim.sh");
        parse(line, largv);
        fd = open("out_sim.txt", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        dup2(fd, 1); dup2(fd, 2);  // make stdout and stderr go to file
        close(fd);
        execvp(*largv, largv);
    }
    pthread_mutex_unlock(&fork_mutex);
    waitpid(rf, &status, 0);
    sync();
    chdir(cwd);
}

void rm_files_from_taskdir(char *taskdir) {
    char cwd[BUF_LEN];

    getcwd(cwd, sizeof(cwd));
    chdir(taskdir);
    rmrf("ply");
    rmrf("test");
    rmrf("rbcs-ic.txt");
    rmrf("load_modules.sh");
    chdir(cwd);
}

#if 0
void post_task(char *taskdir, double *sim_data) {
    int rf, fd, i, status;
    char line[BUF_LEN], *largv[2], fname[BUF_LEN], cwd[BUF_LEN];
    FILE *fp;

    getcwd(cwd, sizeof(cwd));
    chdir(taskdir);
    snprintf(fname, BUF_LEN, "%s/out_post.txt", taskdir);
    rf = fork();
    if (rf < 0) {
        printf("spawner(%d): fork failed!\n", getpid()); fflush(0);
    } else if (rf == 0) {
        snprintf(line, BUF_LEN, "./post.py");
        fd = open(fname, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        dup2(fd, 1); dup2(fd, 2);  // make stdout and stderr go to file
        close(fd);
        parse(line, largv);
        execvp(*largv, largv);
    }
    waitpid(rf, &status, 0);
    sync();

    fp = fopen(fname, "r");
    if (!fp) {
        printf("Cannot open file %s\n", fname); fflush(0);
        abort();
    }
    for (i = 1; i < sim_d; ++i) {
        if (fscanf(fp, "%lf", &sim_data[i]) != 1) {
            printf("Cannot read data point %d from file %s\n", i, fname);
            fflush(0);
            abort();
        }
        if (debug) printf("Output: %d %.6lf\n", i, sim_data[i]);
    }
    sim_data[1] *= sim_data[0];
    sim_data[2] *= sim_data[0];
    fclose(fp);
    chdir(cwd);
    if (!debug) rm_files_from_taskdir(taskdir);
}
#endif

void fitfuntask(const char *fitdir, double *par, int *n, double *sim_data) {
    char taskdir[BUF_LEN];
    snprintf(taskdir, BUF_LEN, "%s/sh_%d", fitdir, (int)sim_data[0]);
    mkdir(taskdir, S_IRWXU);
    write_task_file(taskdir, par, *n);
    pre_task(taskdir, par, *n);
    run_task(taskdir, par, *n);
}

void print_summary_start(double *input, int n, char *fitdir) {
    int i;
    if (display) {
        printf("Starting in %s: ", fitdir);
        for (i = 0; i < n-1; i++) printf("%.6lf ", input[i]);
        printf("%.6lf\n", input[n-1]);
        fflush(0);
    }
}

void write_parfile(char *fitdir, double *input, int n) {
    int i;
    char fname[BUF_LEN];
    snprintf(fname, BUF_LEN, "%s/params.txt", fitdir);
    FILE *fp = fopen(fname, "w");
    if (!fp) {
        printf("Cannot open file %s\n", fname); fflush(0);
        abort();
    }
    for (i = 0; i < n; i++) fprintf(fp, "%.16lf\n", input[i]);
    fclose(fp);
}

void run_all_tasks(const char *fitdir, double *input, int n, double *sim_data) {
    int i;
    double par[n];
    char dirname[BUF_LEN];

    strncpy(dirname, fitdir, BUF_LEN);
    for (i = 0; i < sim_n; ++i) S(i, 0) = i+1;  // shear rates (model units)
    for (i = 0; i < n; ++i) par[i] = input[i];  // prepare input
    for (i = 0; i < sim_n; ++i) {
        torc_create(-1, fitfuntask, 4,
                // lengths and types
                strlen(dirname), MPI_INT,    CALL_BY_COP,
                n,               MPI_DOUBLE, CALL_BY_COP,
                1,               MPI_INT,    CALL_BY_COP,
                sim_d,           MPI_DOUBLE, CALL_BY_RES,
                // arrays
                dirname, par, &n, &S(i, 0));
    }
#ifdef _STEALING_
    torc_enable_stealing();
#endif
    torc_waitall();
#ifdef _STEALING_
    torc_disable_stealing();
#endif
}

void write_resfile(const char *fitdir, double *sim_data) {
    int i, j;
    char fname[BUF_LEN];
    FILE *fp;

    snprintf(fname, BUF_LEN, "%s/result.txt", fitdir);
    fp = fopen(fname, "w");
    if (!fp) {
        printf("Cannot open file %s\n", fname); fflush(0);
        abort();
    }
    for (i = 0; i < sim_n; ++i) {
        for (j = 0; j < sim_d; ++j) fprintf(fp, "%lf ", S(i, j));
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void copy_output(double *output, double *sim_data) {
    int i, j;
    if (output)
        for (i = 0; i < sim_n; ++i)
            for (j = 0; j < sim_d; ++j)
                output[i*sim_d+j] = S(i, j);
}

void plot_fit(char *fitdir) {
    int rf, status;
    char line[BUF_LEN], *largv[BUF_LEN], cwd[BUF_LEN];

    getcwd(cwd, sizeof(cwd));
    chdir(fitdir);
    rf = fork();
    if (rf < 0) {
        printf("spawner(%d): fork failed!\n", getpid()); fflush(0);
    } else if (rf == 0) {
        snprintf(line, BUF_LEN, "./plot.gp");
        parse(line, largv);
        execvp(*largv, largv);
    }
    waitpid(rf, &status, 0);
    sync();
    chdir(cwd);
}

double compute_fitness(char *fitdir, double *sim_data, double sigma) {
    int i, j, rf, fd, status; 
    char line[BUF_LEN], *largv[2], cwd[BUF_LEN];

    getcwd(cwd, sizeof(cwd));
    chdir(fitdir);

    FILE *fp = fopen("sim_data.txt", "w");
    if (!fp) {
        printf("Can't create file.\n"); fflush(0);
        abort();
    }
    for (i = 0; i < sim_n; ++i) {
        for (j = 0; j < sim_d; ++j)
            fprintf(fp, "%.16lf ", S(i,j));
        fprintf(fp, "\n");
    }

    while (pthread_mutex_trylock(&fork_mutex) == EBUSY) usleep(1e6);
    rf = fork();
    if (rf < 0) {
        printf("Fork failed\n"); fflush(0);
    } else if (rf == 0) {
        sprintf(line, "fit.py");
        parse(line, largv);
        fd = open("out_fit.txt", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        dup2(fd, 1); dup2(fd, 2);  // make stdout and stderr go to file
        close(fd);
        execvp(*largv, largv);
    }
    pthread_mutex_unlock(&fork_mutex);
    waitpid(rf, &status, 0);
    sync();
    chdir(cwd);
#if 0
    int i, j;
    double s2, d2, ss2, se2, loglik = 0, sigma2 = pow(sigma, 2);
    double sim_data[sim_n*sim_d];

    // TODO: fit sim data with f(x) = a+(1-a)*exp(-t*x)

    for (i = 0; i < exp_n; ++i)
        for (j = 1; j < exp_d; j += 2) {
            ss2 = pow(S(i, j+1), 2);
            se2 = pow(E(i, j+1), 2);
            s2 = sigma2 + ss2 + se2;
            d2 = pow(E(i, j) - S(i, j), 2);
            loglik += log(s2) + d2/s2;
        }
    loglik += exp_n*log(2*M_PI);
    loglik *= -0.5;

    return loglik;
#endif
}

void print_summary_end(double *input, int n, char *fitdir, double res, double t) {
    int i;
    if (display) {
        printf("Finished in %s: ", fitdir);
        for (i = 0; i < n-1; i++) printf("%.16lf ", input[i]);
        printf("%.16lf = %.16lf in %g secs\n", input[n-1], res, t);
        fflush(0);
    }
}

double fitfun(double *input, int n, void *output, int *info) {
    double t, p, res = fail, sigma = input[n-1];
    int i, ok = 1;
    char cwd[BUF_LEN], fitdir[BUF_LEN];
    double sim_data[sim_n*sim_d];

    if (n != 2) {
        printf("There should be 2 parameters: gammaC and sigma. Exiting.\n");
        abort();
    }

    getcwd(cwd, sizeof(cwd));
    snprintf(fitdir, BUF_LEN, "%s/tmpdir.%d.%d.%d.%d", cwd,
             info[0], info[1], info[2], info[3]);
    mkdir(fitdir, S_IRWXU);
    print_summary_start(input, n, fitdir);

    t = my_gettime();
    write_parfile(fitdir, input, n);
    run_all_tasks(fitdir, input, n, sim_data);
    write_resfile(fitdir, sim_data);
    copy_output(output, sim_data);
    t = my_gettime() - t;

    /* check for NaN and Inf */
    for (i = 0; i < sim_n*sim_d; ++i) {
        p = sim_data[i];
        if (isnan(p) || isinf(p) || p <= 1e-12) ok = 0;
    }

    if (ok) {
        res = compute_fitness(fitdir, sim_data, sigma);
    } else {
        res = fail;
    }

    print_summary_end(input, n, fitdir, res, t);
    return res;
}
