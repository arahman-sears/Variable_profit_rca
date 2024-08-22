import pandas as pd
import snowflake.connector
from datetime import datetime,timedelta
import pickle
import os
import copy
#load_dotenv()

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
    with ftbl as(
 SELECT * ,  
    row_number() over(partition by  (base.svc_un_no ||'-'||base.so_no) order by base.SO_STS_DT desc) as rn,
    Case when left(base.so_sts_desc,2) in ('CO','ED') then 1
    else 0
    end as completes
    FROM FINANCE_ANALYTICS.ADHOC_TBLS.IHR_UNIT_ECONOMICS AS base
    WHERE  completes =1 

),
ft as (select *
from ftbl
where rn=1),
gr_stats as (
 SELECT 
         SO_STS_DT,
        AVG(variable_profit) AS dt_avg_pr
    FROM ft
    WHERE SO_STS_DT >= DATEADD(year, -1, CURRENT_DATE)  and SO_STS_DT < CURRENT_DATE()
        AND completes=1
    group by SO_STS_DT
    order by SO_STS_DT 
    desc
),
stats AS (
    SELECT 
        AVG(dt_avg_pr) AS mean_profit, 
        STDDEV(dt_avg_pr) AS stddev_profit
    from gr_stats
),

-- Step 2: Find records within the last 4 days where variable_profit is less than 2 stddev below the mean
recent_data AS (
    SELECT 
         
        SO_STS_DT,
        dt_avg_pr
    FROM gr_stats
    WHERE SO_STS_DT >= DATEADD(day, -30, CURRENT_DATE)  and SO_STS_DT <CURRENT_DATE()
       
)
-- select * from gr_stats

-- Step 3: Combine the two results to filter based on the condition
SELECT 
    r.SO_STS_DT,
    r.dt_avg_pr,
    s.mean_profit,
    s.stddev_profit
FROM 
    recent_data r, 
    stats s
WHERE 
    r.dt_avg_pr <(s.mean_profit - 2 * s.stddev_profit)
    
order by DT_AVG_PR desc
    """

    df = pd.read_sql(query, conn)
    if df.shape[0]>0:
      dic=df.iloc[0].to_dict()
      dic['SO_STS_DT']=str(dic['SO_STS_DT'])
      return dic
    else:
      return {}


def join_and_select_given_date(conn, dim,dim_dic, fc, date_col,date_val):
    select_clause = ", ".join([f"{key} AS {value}" for key, value in dim_dic.items()])
    select_clause += ", " + ", ".join(fc)

    
    query = f"""
    WITH nt AS (
        SELECT {select_clause},  
        ROW_NUMBER() OVER(PARTITION BY (SVC_UN_NO || '-' || SO_NO) ORDER BY SO_STS_DT DESC) AS rn,
        CASE 
            WHEN LEFT(SO_STS_DESC, 2) IN ('CO', 'ED') THEN 1
            ELSE 0
        END AS completes
        FROM FINANCE_ANALYTICS.ADHOC_TBLS.IHR_UNIT_ECONOMICS
        WHERE {date_col} = '{date_val}'
        AND completes = 1
    )
    SELECT * 
    FROM nt 
    WHERE rn = 1;
    """

    # Execute the query and get the result in a Pandas DataFrame
    cur = conn.cursor()
    cur.execute(query)
    df = cur.fetch_pandas_all() 
    if 'RN' in df.columns:
        df = df.drop(columns=['RN'])

    for val in dim:
        if val=='ATTEMPTS':
            df['ATTEMPTS'] = df['ATTEMPTS'].apply(lambda x: str(int(x)) if not (pd.isna(x) or None) else '-1')
   

    return df


    return df
def get_all_comb_table(conn):
  query="""select * from IT_ANALYTICS.SHS_PRODUCT.VP_BREAKDOWN"""
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

def ret_vp_cat():
    return {
    'Labor': {
        'Revenue': {
            'LABOR_REVENUE': {}
        },
        'Expense': {
            'LABOR_DIRECT_EXPENSE': {}
        },
        'Discount': {
            'LABOR_DISCOUNT': {}
        }
    },
    'Parts': {
        'Revenue': {
            'PARTS_REVENUE': {},
            'IW_PARTS_REVENUE': {}
        },
        'Expense': {
            'TOTAL_PARTS_COGS_EXPENSE': {}
        },
        'Discount': {
            'PART_DISCOUNT': {}
        }
    },
    'Shipping': {
        'Revenue': {},
        'Expense': {
            'TOTAL_SHIPPING_FORWARD_TRUCK_EXPENSE': {},
            'TOTAL_SHIPPING_FORWARD_CUSTOMER_EXPENSE': {},
            'TOTAL_REVERSE_SHIPPING_DIRECT_EXPENSE': {},
            'TOTAL_REVERSE_SHIPPING_INDIRECT_EXPENSE': {}
        },
        'Discount': {}
    },
    'Truck': {
        'Revenue': {},
        'Expense': {
            'TOTAL_TRUCK_EXPENSE': {}
        },
        'Discount': {}
    },
    'Write-Off': {
        'Revenue': {},
        'Expense': {
            'TOTAL_WRITE_OFF_DIRECT_EXPENSE': {},
            'TOTAL_WRITE_OFF_INDIRECT_UNINSTALL_EXPENSE': {}
        },
        'Discount': {}
    },
    'Marketing': {
        'Revenue': {},
        'Expense': {
            'TOTAL_MARKETING_EXPENSE': {}
        },
        'Discount': {}
    },
    'MSO': {
        'Revenue': {},
        'Expense': {
            'TOTAL_MSO_EXPENSE': {}
        },
        'Discount': {}
    },
    'Credit Card': {
        'Revenue': {},
        'Expense': {
            'CREDIT_CARD_TRANSACTIONAL_EXPENSE': {}
        },
        'Discount': {}
    },
    'Other': {
        'Revenue': {
            'OTHER_REVENUE': {}
        },
        'Expense': {},
        'Discount': {}
    },
    'Home Warranty Reimbursement': {
        'Revenue': {
            'HOME_WARRANTY_REIMBURSEMENT': {}
        },
        'Expense': {},
        'Discount': {}
    }
}
def chek_varprofit(fil_ac_dic,tod_mean,cp_dic):
    mvp=float(fil_ac_dic['MEAN_DAILY_AVG_VARIABLE_PROFIT'])
    svp=float(fil_ac_dic['STD_DAILY_AVG_VARIABLE_PROFIT'])
    dvp=float(tod_mean['VARIABLE_PROFIT'])
    if dvp!=None and svp!=None and dvp!=None:
        return dvp<mvp-cali*svp
    return False

def get_info(fil_ac_dic,tod_mean,cp_dic,vp_cat):
  #print(cali)
  res={}
  pref_mn='MEAN_DAILY_AVG_'
  pref_std='STD_DAILY_AVG_'
  lst=[f'{v} : {cp_dic[v]}' for v in cp_dic if cp_dic[v]!='ALL']
  x=round(tod_mean['VARIABLE_PROFIT'],2)
  y=round(fil_ac_dic[pref_mn+'VARIABLE_PROFIT'],2)
  ret_str=f"The variable profit is $ {x}, which is $ {y-x} below daily average of $ {y} with standard deviation $ {round(fil_ac_dic[pref_std+'VARIABLE_PROFIT'],2)} for breakdown of  " +" and ".join(lst)+" ,making this group an outlier. "
  res['value']=ret_str
  res['node']= ' and '.join(lst)
  den=abs(fil_ac_dic[pref_std+'VARIABLE_PROFIT'])
  if den==0:
    res['z-score']=0
  else:
    res['z-score']=abs(fil_ac_dic[pref_mn+'VARIABLE_PROFIT']-tod_mean['VARIABLE_PROFIT'])/den
  for cat in vp_cat:
    for val in vp_cat[cat]:
      for v in vp_cat[cat][val]:
        print(v)
        delta=tod_mean[v]-fil_ac_dic[pref_mn+v]
        mn=fil_ac_dic[pref_mn+v]
        std=fil_ac_dic[pref_std+v]
        vp_cat[cat][val][v]['delta']=round(delta,3)
        vp_cat[cat][val][v]['mean']=round(mn,3)
        vp_cat[cat][val][v]['std']=round(std,3)
  res['water_fall']=vp_cat

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

def rec(num,lst_key,dim_f,dic,st,df_ac,tod_df,pr_res,vp_cat):
    nd=Node(value=pr_res)

    for i in range(len(lst_key)):
        if not check_nth_bit(num, i):
            nnum=set_nth_bit(num, i)
            for val in dim_f[lst_key[i]]:
                cp_dic=dic.copy()
                cp_dic[lst_key[i]]=val
                if not chec_enter(cp_dic):
                    continue
                tup=tuple(cp_dic.values())
      
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
                        #print('vp',tod_mean['VARIABLE_PROFIT'],fil_ac_dic[0]['MEAN_DAILY_AVG_VARIABLE_PROFIT'],fil_ac_dic[0]['STD_DAILY_AVG_VARIABLE_PROFIT'],cp_dic)
                        vp_cat_cp=copy.deepcopy(vp_cat)
           
                        res=get_info(fil_ac_dic[0],tod_mean,cp_dic,vp_cat_cp)
                        if res['z-score']==0:
                            continue
                        nd.next_nodes.append(rec(nnum,lst_key,dim_f,cp_dic,st,df_ac,tod_df,res,vp_cat))
    if len(nd.next_nodes)!=0:
        nd.sort_next_nodes_by_z_score()
        nd.next_nodes=nd.next_nodes[:min(len(nd.next_nodes),5)]

    return nd


def starting_node(dic):
    res={}

    res['value']=None
    val1=dic['SO_STS_DT']
    val2=round(dic['DT_AVG_PR'],2)
    val3=round(dic['MEAN_PROFIT']-dic['DT_AVG_PR'],2)
    res['node']=f"For the date: {val1}, the daily avg profit is $ {val2} which is $ {val3} less than  mean daily average: $ {round(dic['MEAN_PROFIT'],2)} with standard deviation: ${round(dic['STDDEV_PROFIT'],2)}, making it an outlier"
    res['z-score']=None
    res['water_fall']=None
    return res

def save_node(node, filename):
    with open(filename, 'wb') as file:
        pickle.dump(node, file)
def load_node(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
def tree_gen():
    conn=create_snowflake_connection(os.getenv('SNOWFLAKE_USER'), os.getenv('SNOWFLAKE_PASSWORD'), 'sears_hs_prod.us-east-1', 'HS_SUPCH_ANALYTICS_WH', 'IT_ANALYTICS', 'SHS_PRODUCT')
    filtered_columns= [
    # Revenues
    'LABOR_REVENUE',
    'PARTS_REVENUE',
    'IW_PARTS_REVENUE',
    'OTHER_REVENUE',
    'HOME_WARRANTY_REIMBURSEMENT',

    # Expenses
    'LABOR_DIRECT_EXPENSE',
    'TOTAL_PARTS_COGS_EXPENSE',
    'TOTAL_SHIPPING_FORWARD_TRUCK_EXPENSE',
    'TOTAL_SHIPPING_FORWARD_CUSTOMER_EXPENSE',
    'TOTAL_REVERSE_SHIPPING_DIRECT_EXPENSE',
    'TOTAL_REVERSE_SHIPPING_INDIRECT_EXPENSE',
    'TOTAL_TRUCK_EXPENSE',
    'TOTAL_WRITE_OFF_DIRECT_EXPENSE',
    'TOTAL_WRITE_OFF_INDIRECT_UNINSTALL_EXPENSE',
    'TOTAL_MARKETING_EXPENSE',
    'TOTAL_MSO_EXPENSE',
    'CREDIT_CARD_TRANSACTIONAL_EXPENSE',

    # Discounts
    'LABOR_DISCOUNT',
    'PART_DISCOUNT',

    # Variable Profit
    ]
    filtered_columns = filtered_columns+['VARIABLE_PROFIT']
    dim=[ 'CALL_TYPE','APPLIANCE', 'PLANNING_AREA_NAME', 'ATTEMPTS']
    dim_dic={'HOME_SERVICES_SPECIALITY':'APPLIANCE', 'PLANNING_AREA_NAME':'PLANNING_AREA_NAME','CALL_TYPE': 'CALL_TYPE', 'N_ATP':'ATTEMPTS'}
    faulty_dic=get_faulty_profit_last_30(conn)
    
    dic={'PLANNING_AREA_NAME': 'ALL',
        'CALL_TYPE': 'ALL',
         'APPLIANCE': 'ALL',
         'ATTEMPTS': 'ALL'}
    if  faulty_dic:
        result_df=join_and_select_given_date(conn, dim,dim_dic, filtered_columns, 'SO_STS_DT',faulty_dic['SO_STS_DT'])
        dim_f={v:result_df[v].unique() for v in dim}
        df_ac=get_all_comb_table(conn)
        start_val=starting_node(faulty_dic)
        num=0
        lst_key=[v for v in dim_f]
        #indx_dic={lst_key[i]:i for i in range(len(lst_key))}
        variable_profit_categories=ret_vp_cat()
        res_nd=rec(num,lst_key,dim_f,dic,set(),df_ac,result_df,start_val,variable_profit_categories)
        return res_nd
    else:
        return None
res_nd=tree_gen()
save_node(res_nd, 'tree.pkl')
nn=load_node('tree.pkl')
print(res_nd.next_nodes[-1].value)
print(nn.next_nodes[-1].value)
    