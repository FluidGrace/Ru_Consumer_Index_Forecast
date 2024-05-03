import pandas as pd
import glob
#openpyxl позволяет использовать pandas.read_excel, так что устанавливаем (pip install openpyxl)

if __name__ == "__main__":
    #конец названия файла меняется каждый месяц
    for x in glob.glob('ipc_mes*.xlsx'):
        io = x
       
    #уберём лишние строки и столбцы, оставим только строку на каждый месяц и столбец на каждый год
    def trunc(io, sheetnum):
        xls = pd.read_excel(io, sheet_name=sheetnum)
        head_col = xls.shape[1] - 32
        head_row = 0
        index_row = 0
        for i in range(xls.shape[0]):
            for j in range(head_col):
                cell = xls.iloc[i,j]
                if '1991' in str(cell):
                    head_row = i+1
                    head_col = j
                if 'январь' in str(cell):
                    index_row = i+1
                    break
            if index_row>0:
                break
        return pd.read_excel(io, sheet_name=sheetnum, header=head_row, skiprows=range(head_row+1,index_row),
                            usecols=range(head_col,xls.shape[1]), nrows=12)

    trunc(io,1).to_csv('month-aggregate.csv',index=False)
    trunc(io,2).to_csv('month-food.csv',index=False)
    trunc(io,3).to_csv('month-nonfood.csv',index=False)
    trunc(io,4).to_csv('month-service.csv',index=False)
