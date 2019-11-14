from pandas.tests.io.excel.test_xlsxwriter import xlsxwriter


def create_xls(file_name, column_names, content):
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()

    column_name_index = 0
    for column_name in column_names:
        worksheet.write(0, column_name_index, column_name)
        column_name_index += 1

    row = 0
    column = 0

    for each_row in content:
        for each_value in each_row:
            worksheet.write(row, column, each_value)
            column += 1
        row += 1

    workbook.close()
