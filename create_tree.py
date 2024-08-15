import pandas as pd
import snowflake.connector
from datetime import datetime,timedelta
import pickle
import os
load_dotenv()

cali=2
class Node:
    def __init__(self, value=None, next_nodes=None):
        self.value = value  # Store the value as a dictionary
        self.next_nodes = next_nodes if next_nodes else []  # List of next nodes

    def sort_next_nodes_by_z_score(self):
        self.next_nodes.sort(key=lambda node: node.value['z-score'], reverse=True)

    def __repr__(self):
        return f'Node(value={self.value})'
def create_snowflake_connection(user, password, account, warehouse, database, schema):
    
    """
    Create and return a Snowflake connection.
    """
    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema
    )
    return conn
def get_faulty_profit_last_30(conn):
    
    query = """
    WITH ftbl AS (
        SELECT
            a.svc_un_no,
            a.so_no,
            b.variable_profit,
            a.COMPLETES,
            a.SO_STS_DT,
            ROW_NUMBER() OVER(PARTITION BY (a.svc_un_no || '-' || a.so_no) ORDER BY a.SO_STS_DT DESC) AS rn
        FROM FINANCE_ANALYTICS.DI_PRD_TBLS.FACT__UE_METRICS a
        INNER JOIN FINANCE_ANALYTICS.DI_PRD_TBLS.BASE__IHR_UNIT_ECONOMICS b
        ON a.svc_un_no = b.svc_un_no
        AND a.so_no = b.so_no
    ),
    ft AS (
        SELECT *
        FROM ftbl
        WHERE rn = 1
    ),
    gr_stats AS (
        SELECT
            SO_STS_DT,
            AVG(variable_profit) AS dt_avg_pr
        FROM ft
        WHERE SO_STS_DT >= DATEADD(year, -1, CURRENT_DATE)
        AND SO_STS_DT < CURRENT_DATE()
        AND completes = 1
        GROUP BY SO_STS_DT
        ORDER BY SO_STS_DT DESC
    ),
    stats AS (
        SELECT
            AVG(dt_avg_pr) AS mean_profit,
            STDDEV(dt_avg_pr) AS stddev_profit
        FROM gr_stats
    ),
    recent_data AS (
        SELECT
            SO_STS_DT,
            dt_avg_pr
        FROM gr_stats
        WHERE SO_STS_DT >= DATEADD(day, -30, CURRENT_DATE)
        AND SO_STS_DT < CURRENT_DATE()
    )
    SELECT
        r.SO_STS_DT,
        r.dt_avg_pr,
        s.mean_profit,
        s.stddev_profit
    FROM
        recent_data r,
        stats s
    WHERE
        r.dt_avg_pr < (s.mean_profit - 2 * s.stddev_profit)
    ORDER BY dt_avg_pr DESC
    """

    df = pd.read_sql(query, conn)
    if df.shape[0]>0:
      dic=df.iloc[0].to_dict()
      dic['SO_STS_DT']=str(dic['SO_STS_DT'])
      return dic
    else:
      return {}
def get_filtered_columns(conn):
    query = """
    select * from FINANCE_ANALYTICS.DI_PRD_TBLS.BASE__IHR_UNIT_ECONOMICS limit 1
    """


    # Execute the query and load the results into a Pandas DataFrame
    df = pd.read_sql(query, conn)

    # Close the connection
    # conn.close()

    # Filter columns that end with 'EXPENSE' or 'REVENUE' and convert to uppercase
    filtered_columns = [col.upper() for col in df.columns if col.upper().endswith('EXPENSE') or col.upper().endswith('REVENUE')]

    return filtered_columns
def join_and_select_given_date(conn, dim, kpi, date_col,date_val):
    query1= """
    select * from FINANCE_ANALYTICS.DI_PRD_TBLS.BASE__IHR_UNIT_ECONOMICS limit 1
    """
    df_ue = pd.read_sql(query1, conn)
    query2= """
    select * from FINANCE_ANALYTICS.DI_PRD_TBLS.FACT__UE_METRICS limit 1
    """
    df_met = pd.read_sql(query2, conn)
    quaified_columns=[]
    for c in dim:
      if c in df_ue.columns:
        quaified_columns.append(f"base.{c}")
      elif c in df_met.columns:
        quaified_columns.append(f"fact.{c}")

    for c in kpi:
      if c in df_ue.columns:
        quaified_columns.append(f"base.{c}")
      elif c in df_met.columns:
        quaified_columns.append(f"fact.{c}")
    if date_col in df_ue.columns:
        quaified_columns.append(f"base.{date_col}")
        date_col=f"base.{date_col}"
    elif date_col in df_met.columns:
        quaified_columns.append(f"fact.{date_col}")
        date_col=f"fact.{date_col}"


    # Build the SQL query
    query = f"""
    with nt as(
    SELECT {', '.join(quaified_columns)} , row_number() over(partition by  (fact.svc_un_no ||'-'||fact.so_no) order by fact.SO_STS_DT desc) as rn
    FROM FINANCE_ANALYTICS.DI_PRD_TBLS.BASE__IHR_UNIT_ECONOMICS AS base
    INNER JOIN FINANCE_ANALYTICS.DI_PRD_TBLS.FACT__UE_METRICS AS fact
    ON base.svc_un_no = fact.svc_un_no AND base.so_no = fact.so_no
    WHERE {date_col} = '{date_val}' and fact.COMPLETES is not NULL )
    SELECT * from nt where rn=1

    """

    # Execute the query and load the results into a Pandas DataFrame
    df = pd.read_sql(query, conn)
    for val in dim:
      if val=='ATTEMPTS':
        df['ATTEMPTS'] = df['ATTEMPTS'].apply(lambda x: str(int(x)) if not (pd.isna(x) or None) else '-1')
      # else:
      #   df[val] = df[val].apply(lambda x: x.fillna("None"))


    return df
def get_all_comb_table(conn):
    
    query="""select * from IT_ANALYTICS.SHS_PRODUCT.ALL_COMBINATION_PROFIT"""
    df=pd.read_sql(query, conn)
    return df
def check_nth_bit(number, n):
    
    # Shift the bit at position n to the rightmost position
    shifted = number >> n

    # Check if the least significant bit (rightmost) is 1
    return shifted & 1 == 1

def set_nth_bit(number, n):
    mask = 1 << n

    # Use bitwise OR to set the nth bit to 1
    return number | mask

def filter_ac(df,dic):
    df = df[df['APPLIANCE'] == dic['APPLIANCE']]

    df= df[df['PLANNING_AREA_NAME'] == dic['PLANNING_AREA_NAME']]

    df = df[df['CALL_TYPE'] == dic['CALL_TYPE']]

    df = df[df['ATTEMPTS'] == dic['ATTEMPTS']]

    return df.to_dict(orient='records')
def filter_and_average(df: pd.DataFrame, filter_dict: dict) -> dict:
    
    
    # Filter the DataFrame based on the filter_dict, ignoring columns where value is 'ALL'
    #print(df.shape)
    for col, value in filter_dict.items():
        if value != 'ALL':
            #print(col,value)
            df = df[df[col] == value]
            #print(col,value,df.shape)

    # Drop the columns used for filtering where value is not 'ALL'
    columns_to_drop = [col for col, value in filter_dict.items() if value != 'ALL']
    df_filtered = df.drop(columns=columns_to_drop)
    #print(df_filtered.shape)

    # Drop non-numeric columns (int, float, etc.)
    df_filtered = df_filtered.select_dtypes(include=['int', 'float', 'number'])
    #print(df_filtered.shape)
    df_filtered = df_filtered.fillna(0)
    # Check if the DataFrame is empty after filtering
    if df_filtered.empty:
        return {}

    mean_values = df_filtered.mean().to_dict()

    return mean_values


def chek_varprofit(fil_ac_dic,tod_mean,cp_dic):
    mvp=float(fil_ac_dic['MEAN_DAILY_AVG_VARIABLE_PROFIT'])
    svp=float(fil_ac_dic['STD_DAILY_AVG_VARIABLE_PROFIT'])
    dvp=float(tod_mean['VARIABLE_PROFIT'])
    if dvp!=None and svp!=None and dvp!=None:
        return dvp<mvp-cali*svp
    return False

def get_info(fil_ac_dic,tod_mean,cp_dic):
    
    res={}
    pref_mn='MEAN_DAILY_AVG_'
    pref_std='STD_DAILY_AVG_'
    lst=[f'{v} : {cp_dic[v]}' for v in cp_dic if cp_dic[v]!='ALL']
    x=round(tod_mean['VARIABLE_PROFIT'],2)
    y=round(fil_ac_dic[pref_mn+'VARIABLE_PROFIT'],2)
    ret_str=f'Cause for lower average profit $ {x} than mean daily average of $ {y},for the group of '+' and '.join(lst)+':'+'\n'
    res['node']= ' and '.join(lst)
    den=abs(fil_ac_dic[pref_std+'VARIABLE_PROFIT'])
    if den==0:
        den=1.0
    res['z-score']=abs(fil_ac_dic[pref_mn+'VARIABLE_PROFIT']-tod_mean['VARIABLE_PROFIT'])/den
    for v in tod_mean:
        if v not in cp_dic:
            if pref_mn+v not in fil_ac_dic:
                continue
            mkpi=fil_ac_dic[pref_mn+v]
   
            skpi=fil_ac_dic[pref_std+v]
            dkpi=tod_mean[v]
     
            if v.endswith('EXPENSE'):
                if dkpi>mkpi+cali*skpi:
                    var2=dkpi-mkpi
                    y=round(dkpi,2)
                    z=round(var2,2)
                    ret_str+=f'.The average daily expense for {v} is $ {y} ,which is  $ {z}  more than mean daily average \n'
            elif v.endswith('REVENUE'):
                if dkpi<mkpi-cali*skpi:
                    var2=mkpi-dkpi
                    y=round(dkpi,2)
                    z=round(var2,2)
                    ret_str+=f'.The average daily revenue for {v} is $ {y} ,which is $ {z}  less than mean daily average \n'
      # elif v.endswith('COMPLETES'):
      #     if dkpi<mkpi-cali*skpi:
      #       var2=mkpi-dkpi
      #       y=round(dkpi,2)
      #       z=round(var2,2)
      #     ret_str+=f'Average {v} for this breakdown is {y} ,which is {z}  less than mean daily average \n'

    res['value']=ret_str
    return res

def chec_enter(dic):
    for v in dic:
        if not dic[v]:
            return False
        if v=='ATTEMPTS' and dic[v]!='ALL' :
            if not dic[v].isdigit():
                return False
            if int(dic[v])<=0:
                return False

    return True
def rec(num,lst_key,dim_f,dic,st,df_ac,tod_df,pr_res):
    
    nd=Node(value=pr_res)
    print(bin(num)[2:],pr_res['value'])
    for i in range(len(lst_key)):
        if not check_nth_bit(num, i):
            nnum=set_nth_bit(num, i)
            for val in dim_f[lst_key[i]]:
                cp_dic=dic.copy()
                cp_dic[lst_key[i]]=val
                if not chec_enter(cp_dic):
                    continue
                tup=tuple(cp_dic.values())
        # if tup in st:
        #   #print("Already_there",cp_dic)
                if tup not in st:
                    st.add(tup)
                    tod_df_cp=tod_df.copy(True)
                    tod_mean=filter_and_average(tod_df_cp,cp_dic)
                    if len(tod_mean.items())==0:
                        continue
                    df_ac_cp=df_ac.copy(deep=True)
                    fil_ac_dic=filter_ac(df_ac_cp,cp_dic)
                    if len(fil_ac_dic)==0:
                        continue

                    del df_ac_cp
                    del tod_df_cp
                    if chek_varprofit(fil_ac_dic[0],tod_mean,cp_dic):
                        res=get_info(fil_ac_dic[0],tod_mean,cp_dic)
                        nd.next_nodes.append(rec(nnum,lst_key,dim_f,cp_dic,st,df_ac,tod_df,res))
    if len(nd.next_nodes)!=0:
        nd.sort_next_nodes_by_z_score()
    return nd


def starting_node(dic):
    res={}
    res['value']=None
    val1=dic['SO_STS_DT']
    val2=round(dic['DT_AVG_PR'],2)
    val3=round(dic['MEAN_PROFIT']-dic['DT_AVG_PR'],2)
    res['node']=f'For the date: {val1}, the daily avg profit is $ {val2} which is $ {val3} less than  mean daily average, an outlier'
    res['z-score']=None
    return res

def save_node(node, filename):
    with open(filename, 'wb') as file:
        pickle.dump(node, file)
def load_node(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
def tree_gen():
    conn=create_snowflake_connection(os.getenv('SNOWFLAKE_USER'), os.getenv('SNOWFLAKE_PASSWORD'), 'sears_hs_prod.us-east-1', 'HS_SUPCH_ANALYTICS_WH', 'IT_ANALYTICS', 'SHS_PRODUCT')
    kpi=get_filtered_columns(conn)+['COMPLETES']+['VARIABLE_PROFIT']
    dim=['APPLIANCE', 'PLANNING_AREA_NAME', 'CALL_TYPE', 'ATTEMPTS']
    faulty_dic=get_faulty_profit_last_30(conn)
    result_df=join_and_select_given_date(conn, dim, kpi, 'SO_STS_DT',faulty_dic['SO_STS_DT'])
    dim_f={v:result_df[v].unique() for v in dim}
    df_ac=get_all_comb_table(conn)
    dic={'PLANNING_AREA_NAME': 'ALL',
        'CALL_TYPE': 'ALL',
         'APPLIANCE': 'ALL',
         'ATTEMPTS': 'ALL'}
    if  faulty_dic:
        start_val=starting_node(faulty_dic)
        num=0
        lst_key=[v for v in dim_f]
        indx_dic={lst_key[i]:i for i in range(len(lst_key))}
        res_nd=rec(0,lst_key,dim_f,dic,set(),df_ac,result_df,start_val)
        return res_nd
    else:
        None
res_nd=tree_gen()
save_node(res_nd, 'tree.pkl')
nn=load_node('tree.pkl')
print(res_nd.next_nodes[-1].value)
print(nn.next_nodes[-1].value)
    