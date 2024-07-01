#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <locale.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <cassert>
#include <mkl_cblas.h>
#include <mkl.h>
#include <mkl_blas.h>
#include <chrono>
#define N 100

using namespace std;
double GetNormalizedRandomNumber(int digits);
double round(double x, int precision);
void MatrixMult(double* a, double* b, double* c, int n);
int main()
{
	setlocale(LC_ALL, "Rus");
	cout << "Выполнила Самаркина Евгения Андреевна - ФИТУ 09.03.01 ПОВа-з22" << endl;
	srand(time(NULL));
	unsigned int start = 0;
	unsigned int end = 0;
	unsigned int work_time = 0;
	double min[3], max[3];
	// 1й способ
	cout << "Перемножение матриц с использованием формул линейной алгебры: \n " << endl;
	vector<double> Row1 = vector<double>(1024);
	vector<vector<double>> Matrix = vector<vector<double>>(1024);
	vector<vector<double>> Matrix2 = vector<vector<double>>(1024);
	for (int i = 0; i < Row1.capacity(); i++) {
		Row1 = vector<double>(1024);
		for (int j = 0; j < Row1.capacity(); j++) {
			Row1[j] = GetNormalizedRandomNumber(2);
		}
		Matrix[i] = Row1;
		Row1.clear();
	}
	for (int i = 0; i < Row1.capacity(); i++) {
		Row1 = vector<double>(1024);
		for (int j = 0; j < Row1.capacity(); j++) {
			Row1[j] = GetNormalizedRandomNumber(2);
		}
		Matrix2[i] = Row1;
		Row1.clear();
	}

	vector<vector<double>> Result = vector<vector<double>>(1024);
	Row1 = vector<double>(1024);
	for (int j = 0; j < Row1.capacity(); j++) {
		Row1[j] = 0;
	}
	for (int i = 0; i < Row1.capacity(); i++) {
		Result[i] = Row1;
	}

	for (int i = 0; i < 1024; i++) {
		for (int j = 0; j < 1024; j++) {
			for (int k = 0; k < 1024; k++) {
				Result[i][j] += Matrix[i][k] * Matrix2[k][j];
			}
		}
	}
	min[0] = Result[0][0];
	for (int i = 0; i < 1024; i++) {
		for (int j = 0; j < 1024; j++) {
			if (Result[i][j] < min[0]) min[0] = Result[i][j];
		}
	}
	max[0] = Result[0][0];
	for (int i = 0; i < 1024; i++) {
		for (int j = 0; j < 1024; j++) {
			if (Result[i][j] > max[0]) max[0] = Result[i][j];
		}
	}
	cout << "Мнимальный элемент результирующей матрицы: " << min[0] << ", максимальный: " << max[0] << endl;
	cout << endl << "---------------------------------------------------------------" << endl;
	// 2й способ
	cout << "Перемножение матриц с использованием функции cblas_dgemm из библиотеки MKL: \n " << endl;
	int m = 1024, n = 1024, k = 1024, a = 0;
	long long int difficulty = 2 * pow(m * n, 3);
	cout << "Сложность алгоритма: " << difficulty << endl;
	double* A = new double[m * k];

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			A[a] = Matrix[i][j];
			a++;

		}
	}

	a = 0;

	double* B = new double[k * n];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			B[a] = Matrix2[i][j];
			a++;
		}
	}

	double* C = new double[m * n];
	start = clock();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C, n);
	end = clock();
	work_time = end - start;

	cout << endl;
	cout << "Время работы алгоритма из библиотеки mkl: " << (double)work_time / 1000 << " секунд." << endl;
	cout << "Производительность алгоритма: " << endl;
	cout << difficulty / ((double)work_time / 1000 * pow(10, -6)) << endl;
	min[1] = C[0];
	max[1] = C[0];
	for (int i = 0; i < m * n; i++) {
		if (C[i] < min[1]) min[1] = C[i];
	}
	for (int i = 0; i < m * n; i++) {
		if (C[i] > max[1]) max[1] = C[i];
	}
	cout << "Мнимальный элемент результирующей матрицы: " << min[1] << ", максимальный: " << max[1] << endl;
	cout << endl << "---------------------------------------------------------------" << endl;
	start = 0, end = 0;
	// 3й способ
	cout << "Перемножение матриц с помощью алгоритма блочного перемножения матриц: \n " << endl;
	C = new double[m * n];
	auto start1 = chrono::high_resolution_clock::now();
	MatrixMult(A, B, C, n);
	auto end1 = chrono::high_resolution_clock::now();
	auto elapsed_ms = chrono::duration_cast<chrono::microseconds>(end1 - start1);

	cout << endl;
	cout << "Время работы алгоритма блочного перемножения матриц: " << (double)elapsed_ms.count() / 1000000 << " секунд." << endl;
	cout << "Производительность алгоритма: " << endl;
	cout << difficulty / ((double)elapsed_ms.count() / 1000000 * pow(10, -6)) << endl;

	min[2] = C[0];
	max[2] = C[0];
	for (int i = 0; i < m * n; i++) {
		if (C[i] < min[2]) min[2] = C[i];
	}
	for (int i = 0; i < m * n; i++) {
		if (C[i] > max[2]) max[2] = C[i];
	}
	cout << "Мнимальный элемент результирующей матрицы: " << min[2] << ", максимальный: " << max[2] << endl;
	cout << endl << "---------------------------------------------------------------" << endl;
	max[0] = round(max[0], 3);
	max[1] = round(max[1], 3);
	max[2] = round(max[2], 3);
	min[0] = round(min[0], 3);
	min[1] = round(min[1], 3);
	min[2] = round(min[2], 3);
	if ((min[0] == min[1]) && (min[1] == min[2]) && (max[0] == max[1]) && (max[1] == max[2])) cout << "перемножение всеми способами выполнено верно!" << endl;
	else cout << "произошла ошибка в 1 из алгоритмов" << endl;
}

double GetNormalizedRandomNumber(int digits)
{
	int precision = pow(10, digits);
	assert(precision < RAND_MAX); // Precision is too high.
	double ret = rand() % precision;
	ret /= precision;
	return ret;
}
double round(double x, int precision)
{
	int mul = 10;

	for (int i = 0; i < precision; i++)
		mul *= mul;
	if (x > 0)
		return floor(x * mul + .5) / mul;
	else
		return ceil(x * mul - .5) / mul;
}

void MatrixMult(double* a, double* b, double* c, int n) {
	int bm, bi, nbm, nbi;
	int l, nl;
	int i, j, m;
	double* pa, * pb, * pc;
	double s00, s01, s10, s11;
	for (bm = 0; bm < n; bm += N) {
		nbm = (bm + N <= n ? bm + N : n);
		for (bi = 0; bi < n; bi += N) {
			nbi = (bi + N <= n ? bi + N : n);
			for (m = bm, pc = c + bm; m < nbm; m++, pc++) {
				for (i = bi; i < nbi; i++)
					pc[i * n] = 0;
			}
			for (l = 0; l < n; l += N) {
				nl = (l + N <= n ? l + N : n);
				for (m = bm, pc = c + bm; m < nbm; m += 2, pc += 2)
					for (i = bi, pb = b + m; i < nbi; i += 2) {
						pa = a + l + i * n;
						s00 = s10 = s01 = s11 = 0;
						for (j = l; j < nl; j++, pa++) {
							s00 += pa[0] * pb[j * n];
							s01 += pa[0] * pb[j * n + 1];
							s10 += pa[n] * pb[j * n];
							s11 += pa[n] * pb[j * n + 1];
						}
						pc[i * n] += s00;
						pc[i * n + 1] += s01;
						pc[(i + 1) * n] += s10;
						pc[(i + 1) * n + 1] += s11;
					}
			}
		}
	}
}

