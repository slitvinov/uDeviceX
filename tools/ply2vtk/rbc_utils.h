#ifndef RBC_UTILS_H
#define RBC_UTILS_H
#include <cstdlib>
#include <cstdio>
#include <cstring>

template <typename NType>
long gnp(FILE* fd, long nvar) { /* return a number of points in the file
				   (assuming NType and `NVAR' per lines) */
  long sz;
  fseek(fd, 0L, SEEK_END); sz = ftell(fd); fseek(fd, 0, SEEK_SET);
  return sz / sizeof(NType) / nvar;
}

float env2f(const char* n) { /* read a float from env. */
  char* v_ch = getenv(n);
  if (v_ch == NULL) {
    fprintf(stderr, "(rw) ERROR: environment variable `%s' should be set\n", n);
    exit(2);
  }
  return atof(v_ch);
}

int env2d(const char* n) { /* read an integer from env. */
  char* v_ch = getenv(n);
  if (v_ch == NULL) {
    fprintf(stderr, "(rw) ERROR: environment variable `%s' should be set\n", n);
    exit(2);
  }
  return atoi(v_ch);
}

int env2d_default(const char* n, int def) { /* read an integer from env with default */
  char* v_ch = getenv(n);
  return v_ch == NULL ? def : atoi(v_ch);
}

FILE* safe_fopen(const char* fn, const char *mode) {
  FILE* fd = fopen(fn, mode);
  if (fd == NULL) {
    fprintf(stderr, "(rbc_utils) ERROR: cannot open file: %s\n", fn);
    exit(2);
  }
  return fd;
}

size_t safe_fread(void *ptr, size_t size, size_t nmemb, FILE *stream) {
  size_t rc = fread(ptr, size, nmemb, stream);
  if (rc == 0) {
    fprintf(stderr, "(rbc_utils) ERROR: cannot read data\n");
    exit(2);
  }
  return rc;
}

char* trim(char* s) { /* remove trailing and leading blanks, tabs,
			   new lines */
  /* trailing */
  int n;
  for (n = std::strlen(s)-1; n >= 0; n--)
    if (s[n] != ' ' && s[n] != '\t' && s[n] != '\n')
      break;
  s[n+1] = '\0';

  /* and leading */
  while (s[0] != '\0' && (s[0] == ' ' || s[0] == '\t')) s++;
  return s;
}

#endif
