> # **_VSCODE in WINDOWS_**

## C++ windows

在vscode编辑器中使用Microsoft C++ 编译器和调试

> ### 准备
- 安装`vscode`
- 安装`C++ extension for VS Code.
- 安装`Microfost C++`(MSVC)编译器工具
> - 待定



## C++ Mingw-w64 

这篇教程，将使你使用gcc c++ 编译器和GDB debugger去创建windows下运行的程序

> ### 准备
- 安装`vscode`
- 安装`C++ extension for VS Code.
- 安装Mingw-w64到`路径没有任何空格`的文件夹下，也就是不能安装在默认路径(`C:/Program Files/`），本文假设你安装的路径为`C:\Mingw-w64`
- 把Mingw-w64下bin文件夹的路径添加到windows的`PATH`环境变量下面.
> - 通过windows的搜索工具，键入`cmd`
> - 在命令行窗口中，使用`setx`添加Mingw-w64路径到系统路径，例如
```
setx path "%path%;c:\mingw-w64\x86_64-8.1.0-win32-seh-rt_v6-rev0\mingw64\bin"
```
当然，你也可以用windows自带的界面编辑

> ### 创建工作空间
- 如下代码所示：
```
mkdir projects
cd projects
mkdir helloworld
cd helloworld
code .
```
