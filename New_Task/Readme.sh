cd New_Task/

# Get the dataset
python process_data.py --data_dir raw_data_set/daily --prefix daily --start_date 1995/1/1 --unit D --result_dir processed_data/daily/data_daily.csv
python process_data.py --data_dir raw_data_set/weekly --prefix weekly --start_date 2006/7/7 --unit W --result_dir processed_data/weekly/data_weekly.csv

set PYTHONPATH=../ta/
../
# Begin add indicator
python ../main.py --data_dir processed_data/daily --save_dir result/daily_result.xlsx --indicator all
python ../main.py --data_dir processed_data/weekly --save_dir result/weekly_result.xlsx --indicator all

cd ..
python ../main.py --data_dir processed_data/daily --save_dir result/XLNS_daily_result.xlsx --custom_indicator_fn momentum.rsi --custom_indicator_args close,n=14 --limit_code XLNX
python ../main.py --data_dir processed_data/weekly --save_dir result/XLNS_weekly_result.xlsx --custom_indicator_fn momentum.rsi --custom_indicator_args close,n=14 --limit_code XLNX
