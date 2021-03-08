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

#ifndef CUSTOM_CLIP_PLUGIN_H
#define CUSTOM_CLIP_PLUGIN_H

#include "NvInferPlugin.h"
#include <string>
#include <vector>


using namespace nvinfer1;

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.
//自定义插件需要扩展IPluginV2和IPluginCreator类
//声明新的插件类继承IPluginV2
class ClipPlugin : public IPluginV2
{
//定义相关的公有成员，不同插件基本一致，只是具体实现不一样
public:
    //这里声明的函数的具体实现都在customClipPlugin.cpp中
    //定义相关的构造函数，这里定义两种构造函数，根据输入的参数来判断用哪一个
    ClipPlugin(const std::string name, float clipMin, float clipMax);

    ClipPlugin(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make ClipPlugin without arguments, so we delete default constructor.
    //这个插件实现对数据进行截断的功能，必须要接受参数，默认的构造函数要删除
    ClipPlugin() = delete;
    //这里的noexcept表示该函数不会抛出异常，override表示当前函数重写了基类中的虚函数，而这里的const表示这个函数不会修改数据成员
    //这个函数用来返回相应插件op操作返回的tensor数量
    int getNbOutputs() const noexcept override;
    //这里声明一个函数获取输出的维度
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;
    //初始化
    int initialize() noexcept override;
    //停止
    void terminate() noexcept override;
    //获取工作空间的大小，可以直接引用整个网络的工作空间，也可以自己开辟新的，但是如果是自己开辟的化，注意释放
    size_t getWorkspaceSize(int) const noexcept override { return 0; };
    //batch数据的入队操作，实际插件操作的执行函数
    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) noexcept override;
    //获取序列化的尺寸
    size_t getSerializationSize() const noexcept override;
    //序列化函数，用于将插件相关参数保存到文件
    void serialize(void* buffer) const noexcept override;
    //配置相应的参数和格式等等
    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) noexcept override;
    //判断数据格式是否支持
    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;
    //获取当前plugin的类型
    const char* getPluginType() const noexcept override;
    //获取当前插件类型的版本
    const char* getPluginVersion() const noexcept override;
    //销毁插件
    void destroy() noexcept override;
    //复制当前插件
    nvinfer1::IPluginV2* clone() const noexcept override;
    //设置插件的命名空间
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    //获取插件的命名空间
    const char* getPluginNamespace() const noexcept override;

private:
    //私有成员变量，一般根据插件不同而不同
    //这里有相应的层次名称
    const std::string mLayerName;
    //截断的范围
    float mClipMin, mClipMax;
    //输入数据的尺寸
    size_t mInputVolume;
    //命名空间
    std::string mNamespace;
};
//继承的IPluginCreator类
class ClipPluginCreator : public IPluginCreator
{
public:
    //构造函数
    ClipPluginCreator();
    //获取插件的名称
    const char* getPluginName() const noexcept override;
    //获取插件版本
    const char* getPluginVersion() const noexcept override;
    //获取存储插件参数结构的地址
    const PluginFieldCollection* getFieldNames() noexcept override;
    //创建插件
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    //反序列化插件
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    //设置插件的命名空间
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    //获取插件命名空间
    const char* getPluginNamespace() const noexcept override;

private:
    //存储插件相关参数的结构
    static PluginFieldCollection mFC;
    //相关参数
    static std::vector<PluginField> mPluginAttributes;
    //命名空间
    std::string mNamespace;
};

#endif
