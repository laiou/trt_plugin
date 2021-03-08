/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "customClipPlugin.h"
#include "NvInfer.h"
#include "clipKernel.h"

#include <vector>
#include <cassert>
#include <cstring>

using namespace nvinfer1;

// Clip plugin specific constants
//命名空间
namespace
{
const char* CLIP_PLUGIN_VERSION{"1"};
const char* CLIP_PLUGIN_NAME{"CustomClipPlugin"};
} // namespace

// Static class fields initialization
//静态类字段初始化
//具体的声明参考customClipPlugin.h
PluginFieldCollection ClipPluginCreator::mFC{};
//vector是一个存放各种动态类型的数组
std::vector<PluginField> ClipPluginCreator::mPluginAttributes;
//对定义好的插件通过REGISTER_TENSORRT_PLUGIN进行注册
REGISTER_TENSORRT_PLUGIN(ClipPluginCreator);

// Helper function for serializing plugin
//序列化插件
template<typename T>
//将相应参数写入buffer
void writeToBuffer(char*& buffer, const T& val)
{
    //reinterpret_cast也是用作数据类型转换，参考https://www.cnblogs.com/lsgxeva/p/11005293.html
    *reinterpret_cast<T*>(buffer) = val;
    //移动buffer指针位置
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
//反序列化插件的函数
template<typename T>
//从buffer读取相应的值
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}
//插件类的构造函数之一，接收相应的参数，这两个构造函数实际上和下面的ClipPluginCreator和deserializePlugin中的返回相对应
ClipPlugin::ClipPlugin(const std::string name, float clipMin, float clipMax)
    : mLayerName(name)
    , mClipMin(clipMin)
    , mClipMax(clipMax)
{
}
//插件类的另一个构造函数，用于从序列化保存的文件中构造插件
ClipPlugin::ClipPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    //反序列化插件,读取相应的值
    const char *d = static_cast<const char *>(data);
    //记录初始指针位置
    const char *a = d;
    //从相应的Buffer中获取对应的参数
    mClipMin = readFromBuffer<float>(d);
    mClipMax = readFromBuffer<float>(d);
    //判断相关的指针是否正确
    assert(d == (a + length));
}
//获取插件的类别
const char* ClipPlugin::getPluginType() const noexcept
{
    return CLIP_PLUGIN_NAME;
}
//获取插件的版本
const char* ClipPlugin::getPluginVersion() const noexcept
{
    return CLIP_PLUGIN_VERSION;
}
//获取相应插件OP返回的Tensor数量,对于截断而言，返回就只有一个Tensor
int ClipPlugin::getNbOutputs() const noexcept
{
    return 1;
}
//获取插件的输出维度，根据插件不同而不同，这里的截断不会改变维度，从而直接返回输入维度即可
Dims ClipPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    // Validate input arguments
    assert(nbInputDims == 1);
    assert(index == 0);

    // Clipping doesn't change input dimension, so output Dims will be the same as input Dims
    return *inputs;
}
//初始化插件
int ClipPlugin::initialize() noexcept
{
    return 0;
}
//执行插件操作的队列，实际插件操作执行的函数
int ClipPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) noexcept
{
    int status = -1;

    // Our plugin outputs only one tensor
    //声明存储输出的指针数组
    void* output = outputs[0];

    // Launch CUDA kernel wrapper and save its return value
    //clipInference的具体实现参拷clipKernel.cu
    //加载cuda核心并进行相应计算
    status = clipInference(stream, mInputVolume * batchSize, mClipMin, mClipMax, inputs[0], output);
    //根据这个status判断是否执行正确
    return status;
}
//获取序列化的尺寸
size_t ClipPlugin::getSerializationSize() const noexcept
{
    //这里的clip实际上就两个参数，也就是两个浮点数的大小了
    return 2 * sizeof(float);
}
//序列化插件，将插件相关参数保存到文件中
void ClipPlugin::serialize(void* buffer) const noexcept 
{
    //定位buffer指针
    char *d = static_cast<char *>(buffer);
    //记录初始位置
    const char *a = d;
    //将相应的值写入buffer中
    writeToBuffer(d, mClipMin);
    writeToBuffer(d, mClipMax);
    //判断位置是否正确
    assert(d == a + getSerializationSize());
}
//配置相应的支持格式
void ClipPlugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int) noexcept
{
    // Validate input arguments
    //先判断输出的tensor数量和数据类型以及相关的格式是否正确
    assert(nbOutputs == 1);
    assert(type == DataType::kFLOAT);
    assert(format == PluginFormat::kLINEAR);

    // Fetch volume for future enqueue() operations
    //获取相应输入数据的尺度
    size_t volume = 1;
    for (int i = 0; i < inputs->nbDims; i++) {
        //也就是输入数据各个维度相乘，得到输入数据的数量
        volume *= inputs->d[i];
    }
    mInputVolume = volume;
}
//判断输入数据的格式是否匹配
bool ClipPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    // This plugin only supports ordinary floats, and NCHW input format
    if (type == DataType::kFLOAT && format == PluginFormat::kLINEAR)
        return true;
    else
        return false;
}
//终止插件
void ClipPlugin::terminate() noexcept {}

void ClipPlugin::destroy() noexcept {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* ClipPlugin::clone() const noexcept
{
    auto plugin = new ClipPlugin(mLayerName, mClipMin, mClipMax);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void ClipPlugin::setPluginNamespace(const char* libNamespace) noexcept 
{
    mNamespace = libNamespace;
}

const char* ClipPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
//插件创建类的构造函数
ClipPluginCreator::ClipPluginCreator()
{
    // Describe ClipPlugin's required PluginField arguments
    //通过emplace_back在mPluginAttributes容器尾部添加两个PluginFidle类型的变量，实际上就是存储了clip插件的两个截断参数
    mPluginAttributes.emplace_back(PluginField("clipMin", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("clipMax", nullptr, PluginFieldType::kFLOAT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    //给mFC中的元素赋值，这里能看到mFC中保存了mPluginAttributes的size和相应的数据
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}
//获取相应插件的名称
const char* ClipPluginCreator::getPluginName() const noexcept
{
    return CLIP_PLUGIN_NAME;
}
//获取对应插件的版本
const char* ClipPluginCreator::getPluginVersion() const noexcept
{
    return CLIP_PLUGIN_VERSION;
}
//获取插件创建函数中的存储参数的mFC的地址
const PluginFieldCollection* ClipPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}
//利用相关的参数创建一个插件
IPluginV2* ClipPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    //声明相应的变量
    float clipMin, clipMax;
    //提取保存具体参数位置的指针
    const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    //读取相应的参数字段
    //因为clip的参数只有两个，先判断参数量对不对
    assert(fc->nbFields == 2);
    //循环读取
    for (int i = 0; i < fc->nbFields; i++){
        //将相应的参数进行赋值，利用strcmp依次比较两个字符串中的每一个字符
        //来抽取相应的参数
        if (strcmp(fields[i].name, "clipMin") == 0) {
            //再次确认数据类型
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            //进行赋值，利用static_cast强制转换
            clipMin = *(static_cast<const float*>(fields[i].data));
        } else if (strcmp(fields[i].name, "clipMax") == 0) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            clipMax = *(static_cast<const float*>(fields[i].data));
        }
    }
    //返回一个新的插件类，利用构造函数来创建一个新插件
    return new ClipPlugin(name, clipMin, clipMax);
}
//创建一个插件,反序列化相应的插件
IPluginV2* ClipPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call ClipPlugin::destroy()
    //返回一个新的ClipPlugin，但是这里的参数不同，说白了这里的是从文件中序列化得到的插件数据和相应的数据长度
    return new ClipPlugin(name, serialData, serialLength);
}
//设置插件的命名空间
void ClipPluginCreator::setPluginNamespace(const char* libNamespace) noexcept 
{
    mNamespace = libNamespace;
}
//获取相应插件的命名空间
const char* ClipPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
