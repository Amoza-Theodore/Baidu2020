```python
'''
1. Clean_img
2. Tasking
'''

# Collecting
cd workspace/deepcar/deeplearning_python/src
rm nohup.out
nohup python3 My_Data_Coll.py &

# Debug_Collecting
cd workspace/deepcar/deeplearning_python/src
python3 My_Data_Coll.py

# clean_Data
cd workspace/deepcar/deeplearning_python
rm -rf data
mkdir data
mkdir data/img
```



需求: 

