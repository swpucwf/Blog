# C++学习

## 1. 双引号::的作用

-   // ::代表作用域  如果前面什么都不添加 代表全局作用域

  ```c++
  
  #define  _CRT_SECURE_NO_WARNINGS
  #include <iostream>
  int a = 1000;
  void test01() {
      int  a = 2000;
      // 默认局部作用域
      std::cout << "a = " << a << std::endl;
      // :: 代表全局作用域
      std::cout << "a = " << ::a << std::endl;
  
  };
  
  int main()
  {
      test01();
      std::cout << "Hello World!\n";
  }
  
  
  ```

![image-20240326224714159](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20240326224714159.png)

## 2. namespace命名空间作用

- 命名空间用途： 解决名称冲突

- 命名空间下 可以放  变量、函数、结构体、类...

- 命名空间 必须要声明在全局作用域下
- 命名空间可以嵌套命名空间
- 命名空间是开放的，可以随时给命名空间添加新的成员
- 命名空间可以是匿名的
- 命名空间可以起别名

## 2. using使用原则

- 就近原则；当using声明与 就近原则同时出现，出错，尽量避免。

```c++

#define  _CRT_SECURE_NO_WARNINGS
#include <iostream>

namespace A {
    int a = 100;
}
namespace B {
    int a = 200;
}
void test01() {
    int a = 1;
    std::cout << a << std::endl;
    using namespace A;
    std::cout << a << std::endl;
    using namespace B;
    std::cout << a << std::endl;
}


int main()
{
    test01();
    std::cout << "Hello World!\n";
}


```

![image-20240326225901892](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20240326225901892.png)

-   当using编译指令  与  就近原则同时出现，优先使用就近;当using编译指令有多个，需要加作用域 区分

```c++

#define  _CRT_SECURE_NO_WARNINGS
#include <iostream>

namespace A {
    int a = 100;
}
namespace B {
    int a = 200;
}
void test01() {
    
    using namespace A;
    using namespace B;

    std::cout << A::a << std::endl;
    std::cout << B::a << std::endl;

}


int main()
{
    test01();
    std::cout << "Hello World!\n";
}


```

![image-20240326230113890](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20240326230113890.png)

## 4. c++对C语言的增强

- 全局变量检测增强
- 函数检测增强  返回值没有检测  形参类型没有检测  函数调用参数个数没有检测
- 类型转换检测增强
- 创建结构体变量时候，必须加关键字struct
- struct增强，可以添加函数
- bool类型扩展
- const增强 全局const ，C语言const 修饰默认是外部链接，C++默认是内部链接

## 5. 引用

- 引用必须初始化
- 引用一旦初始化后，就不可以引向其他变量

## 6. 参数传递方式

- 值传递
- 引用传递
- 地址传递

