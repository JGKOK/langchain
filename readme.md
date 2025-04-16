# [基于langchain框架实现简单RAG及聊天历史]

## 概述
基于ubuntu系统、使用ollama大模型使用框架，集成docker容器部署的redis，以langchain框架为基础，实现现简单RAG及聊天历史

## 环境要求
- ubuntu 版本：Ubuntu 22.04
- Python 版本：Python 3.10.12
- Ollama 版本：Ollama 0.6.2
- 依赖库：
  - [langchain]：[0.3.23]
  - [langchain_community]：[0.3.21]

## 安装步骤
1. 根据自己系统版本信息安装对应的ollama版本，对应系统信息查看命令
```shell
  uname -a
```
如果显示x86_64，则前往该地址https://github.com/ollama/ollama/releases，直接下载 ollama-linux-amd64.tgz，如果显示arm字样，则下载 ollama-linux-amd64-rocm.tgz  
2.下载完成后，解压对应安装包
```shell
  tar -C /usr -xzf ollama-linux-amd64-0.6.2.tgz
```
3.创建服务文件并设置自启动
```shell
  vim /etc/systemd/system/ollama.service
```
写入如下内容,设置服务相关信息后保存
```text
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
ExecStart=/usr/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
```
4.生效服务命令并且自启动服务
```shell
  #生效服务命令行
  systemctl daemon-reload
  #服务自启动
  systemctl enable ollama
  #启动服务
  systemctl start ollama
```
5.使用ollama命令检查是否安装成功,安装成功时会显示ollama对应的版本号
```shell
  ollama -v
```
6.使用容器部署redis并且运行容器的redis
```shell
  #启动容器
  docker run -d --name redis-server -p 6379:6379 redis:6.2.16
  #进入容器
  docker exec -it <mycontainer> bash
  #执行redis命令
  redis-cli
```
7.项目结构
1. /doc文件存放训练使用的文档，支持pdf、docx、doc、txt
2. my_rag_history.py主程序运行
3. post_test.py测试API接口调用
    