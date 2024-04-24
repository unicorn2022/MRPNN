#pragma once

#include "vector.cuh"
#include "omp.hpp"

#include <cuda_runtime.h>

#include <vector>
#include <iostream>
using namespace std;

#define HISTO_SIZE 10

struct Histogram {
	float bin[HISTO_SIZE * 3];
	float totalSampleNum;
	float x2;
	float x;
};

#define LUT_SIZE 512

struct Image_host {
private:
	inline float3 Sample(int x, int y) {
		x = x < 0 ? 0 : x;
		x = x > sx - 1 ? sx - 1 : x;
		y = y < 0 ? 0 : y;
		y = y > sy - 1 ? sy - 1 : y;
		return make_float3(data[(y * sx) + x]);
	}
public:
	float4* data = 0;
	int sx, sy;
	float3 Sample(float2 uv) {
		float3 res = make_float3(sx, sy, 0);
		float3 pos = float3{ uv.x,uv.y, 0 } * res - 0.5;
		int x = floor(pos.x);
		int y = floor(pos.y);
		float3 w = pos - make_float3(x, y, 0);
		return lerp(lerp(Sample(x, y), Sample(x + 1, y), w.x), lerp(Sample(x, y + 1), Sample(x + 1, y + 1), w.x), w.y);
	}
};

class VolumeRender {
	/* CPU 端数据 */
	int resolution;		// volume 分辨率
	float* datas;		// volume 数据(CPU), size = resolution^3
	float* hglut;		// LUT 数据(CPU), size = LUT_SIZE^2
	float* mips[9];		// volume mip 数据(CPU), size = (256 >> i)^3
	float* tr_mips[8];	// transmittance mip 数据(CPU), size = (128 >> i)^3
	float* tr_mips2[8];	// 

	/* GPU 端数据 */
	cudaChannelFormatDesc channel_desc;
	cudaExtent size;			// volume 数据(GPU)大小
	cudaArray* datas_dev = 0;	// volume 数据(GPU), size = {resolution, resolution, resolution}
	cudaArray* hglut_dev = 0;	// LUT 数据(GPU), size = {LUT_SIZE, LUT_SIZE}
	cudaExtent mip_size[9];		// volume mip 大小
	cudaArray* mips_dev[9];		// volume mip 数据(GPU), size = {256 >> i, 256 >> i, 256 >> i}
	cudaExtent tr_mip_size[8];	// transmittance mip 大小
	cudaArray* tr_mips_dev[8];	// transmittance mip 数据(GPU), size = {128 >> i, 128 >> i, 128 >> i}
	
	bool checkboard = true;
	bool last_predict = true;

	float hginlut = -100;
	float3 tr_lightDir;
	float tr_alpha;
	float hdri_exp = 1;

	cudaArray* env_tex_dev = 0;

	VolumeRender(const VolumeRender& obj) = delete;

	/* 申请 CPU & GPU 内存 */
	void MallocMemory();

public:
	enum RenderType {
		PT, RPNN, MRPNN
	};

	Image_host hdri_img;

	float max_density = 0.00001;

	/* 初始化 VolumeRender: 分辨率为 resolution */
	VolumeRender(int resolution);
	/* 初始化 VolumeRender: 从 path 中读取 volume */
	VolumeRender(string path);
	/* 销毁 VolumeRender: 释放 CPU & GPU 内存*/
	~VolumeRender();

	void SetData(int x, int y, int z, float value);

	void SetDatas(FillFunc func);
	
	/* 构建mipmap, 将数据拷贝到GPU, 并绑定3D纹理 */
	void Update();
	void Update_TR(float3 lightDir,float alpha = 64.0f, bool CPU = false);

	void SetHDRI(string path);

	void SetCheckboard(bool checkboard);
	void SetEnvExp(float exp);

	void SetTrScale(float scale);
	/* 设置云的散射率 */
	void SetScatterRate(float rate);
	/* 设置云的散射率 */
	void SetScatterRate(float3 rate);
	void SetExposure(float exp);

	void SetSurfaceIOR(float ior);

	float DensityAtPosition(int mip, float3 pos);
	float TrAtPosition(int mip, float3 pos,float3 lightDir);

	float DensityAtPosition(float mip, float3 pos);

	float DensityAtUV(int mip, float3 uv);
	
	float DensityAtUV(float mip, float3 uv);

	void UpdateHGLut(float g);
	
	float GetHGLut(float cos, float angle);
	
	float3 GetTr(float3 ori, float3 dir, float3 lightDir, float alpha = 1, float g = 0, int sampleNum = 1) const;
	
	vector<float3> GetRadiances(vector<float3> ori, vector<float3> dir, float3 lightDir, float3 lightColor = { 1, 1, 1 }, float alpha = 1, int multiScatter = 1, float g = 0, int sampleNum = 1, RenderType rt = RenderType::PT);
	vector<float3> GetSamples(vector<float> alpha, vector<float3> ori, vector<float3> dir, vector<float3> lightDir, vector<float> g, vector<float> scatter, float3 lightColor = { 1, 1, 1 }, int multiScatter = 1, int sampleNum = 1) const;
	vector<float3> GetTrs(float alpha, vector<float3> ori, vector<float3> dir, float3 lightDir, float3 lightColor,float g = 0, int sampleNum = 1) const;

	vector<float3> Render(int2 size, float3 ori, float3 up, float3 right, float3 lightDir, RenderType rt = RenderType::PT, float g = 0.857, float alpha = 1, float3 lightColor = { 1, 1, 1 }, int multiScatter = 512, int sampleNum = 1024);
	void Render(float3* taeget, Histogram* histo_buffer, unsigned int* target2, int2 size, float3 ori, float3 up, float3 right, float3 lightDir, float3 lightColor = { 1,1,1 }, float alpha = 1, int multiScatter = 1, float g = 0, int randseed = 0, RenderType rt = RenderType::PT, int toneType = 2, bool denoise = false);
};