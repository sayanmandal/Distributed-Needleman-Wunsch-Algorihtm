#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>

#define LEN1 100019
#define LEN2 100007
#define match_score 2
#define mismatch_score -1
#define gap_score -2
#define max(a, b) ((a) > (b))?(a):(b)
#define LEFT -1
#define UP 1
#define DIAG 0

int score(char c, char b){
  return (c == b) ? match_score : mismatch_score;
}

char dna[] = {'A', 'G', 'T', 'C'};

void generate(char* c, int length){
  srand((unsigned int)time(NULL));
  for(int i = 0 ; i < length ; i++){
    c[i] = dna[rand()%4];
  }
  c[length] = '\0';
}

int main(int argc, char* argv[]){
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  char* s1 = (char*)malloc((LEN1+1) * sizeof(char));
  char* s2 = (char*)malloc((LEN2+1) * sizeof(char));

  if(rank == 0){

    generate(s1, LEN1);
    generate(s2, LEN2);
    //printf("%s\n", s1);
    //printf("%s\n", s2);
  }
  MPI_Bcast(s1, LEN1+1, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(s2, LEN2+1, MPI_CHAR, 0, MPI_COMM_WORLD);

  int numcols = (LEN1+1)/size;
  int numrows = (LEN2+1);

  int startidx = numcols * rank;

  int* dp = (int*)malloc(2 * numcols * sizeof(int));
  clock_t start, end;
  double cpu_time_used;
//  int* trace = (int*) malloc(numrows * numcols * sizeof(int));

  if(rank == 0){
    start = clock();
  }



  int* x = (int*) malloc(numcols * sizeof(int));
  int* aux = (int*) malloc(numcols * sizeof(int));
  int out;

  for(int i = 0 ; i < numcols ; i++)  dp[i] = gap_score * (startidx+i);

  for(int i = 0 ; i < numcols ; i++)  aux[i] = dp[i];


  if(rank != size-1){
    //printf("there\n");
    MPI_Send(&dp[numcols-1], 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
  }

  int idx1, idx2;
  for(int i = 1 ; i < numrows ; i++){
    idx1 = i%2;
    idx2 = (idx1 + 1)%2;

    dp[idx1 * numcols] = dp[idx2 * numcols] + gap_score;
    for(int j = 1 ; j < numcols ; j++){
      dp[idx1 * numcols + j] = max(dp[idx2 * numcols + j] + gap_score, dp[idx2 * numcols + (j-1)] + score(s1[startidx+j-1], s2[i-1]));
    }

    if(rank != 0){
      //printf("here\n");
      int ret;
      MPI_Recv(&ret, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      dp[idx1 * numcols] = max(dp[idx1 * numcols], ret + score(s1[startidx-1], s2[i-1]));
    }
    /*
    if(rank == 0)
      printf("%d\n", i);
      */

    for(int j = 0 ; j < numcols ; j++){
      dp[idx1 * numcols + j] -= aux[j];
    }

    for(int j = 0 ; j < numcols ; j++)  x[j] = dp[idx1 * numcols + j];



    MPI_Scan(&x[numcols-1], &out, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if(rank != size-1){
      MPI_Send(&out, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
    }


    int pre;
    if(rank != 0){
      MPI_Recv(&pre, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      x[0] = max(x[0], pre);
    }

    for(int j = 1 ; j < numcols ; j++)  x[j] = max(x[j-1], dp[idx1 * numcols + j]);


    for(int j = 0 ; j < numcols ; j++)  dp[idx1 * numcols + j] = x[j] + aux[j];


    if(rank != size-1)
      MPI_Send(&dp[idx1 * numcols + numcols-1], 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);


  }





  if(rank == size-1){
    MPI_Send(&dp[idx1 * numcols + numcols - 1], 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  if(rank == 0){
    int ans;
    MPI_Recv(&ans, 1, MPI_INT, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Score: %d\n", ans);
    end = clock();
    cpu_time_used = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("Time Taken: %.12lf\n", cpu_time_used);
  }



  MPI_Finalize();
  return 0;
}
