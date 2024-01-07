import pandas as pd
import numpy as np
doReframing = True
doCalibrating = False
wave_dict = {
    443: 0,
    514: 1,
    689: 2,
    781: 3,
    817: 4
}
if doReframing:
    df = pd.read_csv('datagroup_12_20.csv')
    rows_as_lists = df.values.tolist()
    headers = rows_as_lists[0]
    columns = ['substrate', 'T_cell', 'T_sub', 'time', 'wavelength', 'TRaw', 'RRaw', 'PRaw', 'T_0','R_0','P_total','R_back','A', 'RT']
    df2 = pd.DataFrame(columns=columns)

    newRows = len(rows_as_lists)//3
    waves = 5

    current_baseline = np.zeros((waves,3)) # TRP
    power_constant = np.zeros((waves,))
    for i in range(newRows):
        l1 = rows_as_lists[3*i] # t
        l2 = rows_as_lists[3*i+1] # r
        l3 = rows_as_lists[3*i+2] # p
        sub = l1[0]
        T_cell = l1[1]
        T_sub = l1[2]
        time = l1[3]
        wavelength = l1[4]
        r_back = l1[6]
        t_raw = l1[7]
        r_raw = l2[7]
        p_raw = l3[7]
        wavedex = wave_dict[int(wavelength)]
        if wavedex == 4:
            continue
        if time < 0:
            current_baseline[wavedex] = [t_raw,r_raw,p_raw]
            power_constant[wavedex] = (r_raw+t_raw)/p_raw
            print(current_baseline)
            print(power_constant)
        else:
            R = r_raw - r_back
            T = t_raw
            RT = R/(R+T)
            P_total = p_raw*power_constant[wavedex]
            print(i)
            print(R)
            A = (P_total - R - T)/P_total
            values = [sub, T_cell,T_sub,time,wavelength,t_raw,r_raw,p_raw,current_baseline[wavedex, 0],current_baseline[wavedex, 1],P_total,r_back,A,RT]
            df2.loc[len(df2)] = values
    print(df2)
    df2.to_csv('datagroup_12_20_reframed.csv', index=False)

if doCalibrating:
    df = pd.read_csv('datagroup_10_05_reframed_no817.csv')
    t_calib = 1000
    df = df[df['time'] >= 1000]
    unique_samples = df['substrate'].unique()
    new_cols = ['443Rc','443Ac','514Rc','514Ac','689Rc','689Ac','781Rc','781Ac']
    ncol_wavindices = [443,443,514,514,689,689,781,781]
    ncol_RAindices = ["RT","A","RT","A","RT","A","RT","A"]
    for i,ncol in enumerate(new_cols):
        df[ncol] = None
        for sub in unique_samples:
            sub_col_val = df[(df['substrate'] == sub) & (df['wavelength'] == ncol_wavindices[i])].iloc[0][ncol_RAindices[i]]
            df.loc[df['substrate'] == sub, ncol] = sub_col_val
    df.to_csv('datagroup_10_05_no817_calib.csv',index=False)

