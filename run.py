import subprocess

# 主函数
def main():
    # 设置一些变量

    # 将变量传递给脚本的方式可以是通过命令行参数或其他方式
    # 运行当前文件夹下的其他脚本
    subprocess.run(["python", "RF.py"])
    subprocess.run(["python", "SVM.py"])
if __name__ == "__main__":
    main()
