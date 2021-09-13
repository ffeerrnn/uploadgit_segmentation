# _*_coding:utf-8_*_
# 作者： 万方名
# 创建日期： 2020/8/26   10:33
# 文件： 2sql.py

sql = ''
task1_result = [{'sel': [1], 'agg': [0], 'cond_conn_op': 2, 'conds': [(3, 0, '11'), (6, 0, '11')]}]
task2_result = {0: {(3, 0, '11'), (6, 0, '11')}}

question_test = 'PE2011大于11或者EPS2011大于11的公司有哪些'
table = {"证券代码": "text", "公司名称": "text", "股价": "real", "EPS2011": "real", "EPS2012E": "real", "EPS2013E": "real",
         "PE2011": "real", "PE2012E": "real", "PE2013E": "real", "NAV": "text", "折价率": "text", "PB2012Q1": "real",
         "评级": "text"}
col_value = {"证券代码": ["600340.SH", "000402.SZ", "600823.SH", "600716.SH", "000608.SZ", "002285.SZ"],
             "公司名称": ['华夏幸福', '金融街', '世茂股份', '凤凰股份', '阳光股份', '世联地产'],
             "股价": ['17.49', '6.53', '11.79', '5.54', '4.79', '15.07'],
             "EPS2011": ['1.54', '0.67', '1.01', '0.32', '0.23', ' 0.48'],
             "EPS2012E": ['2.03', '0.78', '1.13', '0.45', '0.29', '0.84'],
             "EPS2013E": ['2.67', '0.91', '1.39', '0.66', '0.32', '1.05'],
             "PE2011": ['11.36', '9.8', '11.66', '17.51', '20.76', '31.27'],
             "PE2012E": ['8.61', '8.41', '10.4', '12.45', '16.31', '18.04'],
             "PE2013E": ['6.56', '7.2', '8.47', '8.39', '14.75', '14.34'],
             "NAV": ['None', '-38.7', '22.09', '7.46', '6.71', 'None'],
             "折价率": ['None', '-38.7', '-46.6', '-25.8', '-28.7', 'None'],
             "PB2012Q1": ['5.8', '1.0', '1.3', '2.5', '1.4', '3.6'],
             "评级": ['推荐', '谨慎推荐', '无', '谨慎推荐', '谨慎推荐', '无']}
table_name = '表3：2019年第4周（2019.01.28 - 2019.02.03）全国电影票房TOP10'
sample_data = [question_test, table, col_value, table_name]





print(result2sql(task1_result, task2_result, sample_data))
