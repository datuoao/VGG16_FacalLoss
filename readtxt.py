#第一种方法
import re
import openpyxl
import xlwt as xlwt

f = open("D:\code\python\shiyanshi\VGG16_Trans_Body\新建 文本文档.txt","r")   #设置文件对象
line = f.readline()
line = line[:-1]

i = 1
book = xlwt.Workbook(encoding='utf-8')  # 创建Workbook，相当于创建Excel
sheet1 = book.add_sheet(u'Sheet1', cell_overwrite_ok=True)

while line:             #直到读取完文件
    #读取一行文件，包括换行符
    if line[0]=='t' or line[0] =='v':
        # print(line[0:-1])
        sstr = re.findall('\d+', line)  # \d+可匹配一位或多位数字使用
        # print(sstr)
        ss1 = float(sstr[0] + '.' + sstr[1])
        ss2 = float(sstr[2] + '.' + sstr[3])
        # print(ss1)
        # print(type(ss1))
        # print(ss2)

        # 读取到execl
        # 创建sheet，Sheet1为表的名字，cell_overwrite_ok为是否覆盖单元格
        sheet1.write(i, 0, ss1)  # 第0行第0列
        sheet1.write(i,1,ss2)
        i = i+1

    line = f.readline()
    # line = line[:-1]     #去掉换行符，也可以不去

f.close() #关闭文件
book.save("Test.xls") #保存


# import re
#
# str = 'Hello123/World 45_?6bye'
# result1 = re.findall('\d', str)  # \d匹配任何十进制数
# result2 = re.findall('\d+', str)  # \d+可匹配一位或多位数字使用
# result3 = re.findall('\D', str)  # \d匹配非数字字符任何十进制数
# result4 = re.findall('\w', str)  # \w匹配任何字母数字字符，包括下划线在内
# result5 = re.findall('\W', str)  # \W匹配非任何字母数字字符，包括下划线在内
# result6 = re.findall('\s', str)  # \s匹配任何空白字符
# result7 = re.findall('\S', str)  # \S匹配非任何空白字符
# result8 = re.findall('\AHello', str)  # \A仅匹配字符串开头
# result9 = re.findall('bye\Z', str)  # \Z仅匹配字符串结尾
# print(result1)
# print(result2)
# print(result3)
# print(result4)
# print(result5)
# print(result6)
# print(result7)
# print(result8)
# print(result9)







