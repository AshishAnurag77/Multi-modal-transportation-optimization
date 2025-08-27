# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 09:32:04 2018

@author: Ken Huang
"""
# from docplex.mp.model import Model   # ❌ Remove DOCPLEX (needs IBM CPLEX)
from itertools import product
import numpy as np
import cvxpy as cp
import pandas as pd
import json


class MMT:
    '''a Model class that solves the multi-modal transportation optimization problem.'''

    def __init__(self, framework='CVXPY'):   # ✅ default to CVXPY
        # parameters
        self.portSpace = None
        self.dateSpace = None
        self.goods = None
        self.indexPort = None
        self.portIndex = None
        self.maxDate = None
        self.minDate = None
        self.tranCost = None
        self.tranFixedCost = None
        self.tranTime = None
        self.ctnVol = None
        self.whCost = None
        self.kVol = None
        self.kValue = None
        self.kDDL = None
        self.kStartPort = None
        self.kEndPort = None
        self.kStartTime = None
        self.taxPct = None
        self.transitDuty = None
        self.route_num = None
        self.available_routes = None
        # decision variables
        self.var = None
        self.x = None
        self.var_2 = None
        self.y = None
        self.var_3 = None
        self.z = None
        # result & solution
        self.xs = None
        self.ys = None
        self.zs = None
        self.whCostFinal = None
        self.transportCost = None
        self.taxCost = None
        self.solution_ = None
        self.arrTime_ = None
        self.objective_value = None
        # helping variables
        self.var_location = None
        self.var_2_location = None
        self.var_3_location = None

        if framework not in ['CVXPY', 'DOCPLEX']:
            raise ValueError('Framework not supported, the model only supports CVXPY and DOCPLEX')
        else:
            self.framework = framework

    # ------------------------
    # (keep your set_param, build_model, cvxpy_build_model methods same)
    # ------------------------

    def solve_model(self, solver=None):
        '''
        Solve the optimization model & cache the optimized objective value.
        Default solver is ECOS_BB (works in Streamlit Cloud).
        '''
        try:
            if self.framework == 'CVXPY':
                if solver is None:
                    solver = cp.ECOS_BB   # ✅ change default solver to ECOS_BB
                self.objective_value = self.model.solve(solver=solver)
                self.xs = np.zeros((self.portSpace, self.portSpace, self.dateSpace, self.goods))
                self.xs[self.var_location] = self.var.value
                self.ys = np.zeros((self.portSpace, self.portSpace, self.dateSpace))
                self.ys[self.var_2_location] = self.var_2.value
                self.zs = np.zeros((self.portSpace, self.portSpace, self.dateSpace))
                self.zs[self.var_3_location] = self.var_3.value

            # DOCPLEX branch removed for Streamlit simplicity ✅
        except Exception as e:
            raise Exception(f'Model is not solvable: {e}')

        nonzeroX = list(zip(*np.nonzero(self.xs)))
        nonzeroX = sorted(nonzeroX, key=lambda x: x[2])
        nonzeroX = sorted(nonzeroX, key=lambda x: x[3])
        nonzeroX = list(map(lambda x: (self.portIndex[x[0]], self.portIndex[x[1]], \
                                       (self.minDate + pd.to_timedelta(x[2], unit="days")).date().isoformat(),
                                       x[3]), nonzeroX))

        self.whCostFinal, arrTime, _ = self.warehouse_fee(self.xs)
        self.transportCost = np.sum(self.ys * self.tranCost) + np.sum(self.zs * self.tranFixedCost)
        self.taxCost = np.sum(self.taxPct * self.kValue) + \
                       np.sum(np.sum(np.dot(self.xs, self.kValue), axis=2) * self.transitDuty)
        self.solution_ = {}
        self.arrTime_ = {}
        for i in range(self.goods):
            self.solution_[f'goods-{i+1}'] = list(filter(lambda x: x[3] == i, nonzeroX))
            self.arrTime_[f'goods-{i+1}'] = (self.minDate + pd.to_timedelta \
                (np.sum(arrTime[:, self.kEndPort[i], :, i]), unit='days')).date().isoformat()


def transform(filePath):
    '''Read in order and route data, transform the data into a form that can be processed.'''
    order = pd.read_excel(filePath, sheet_name='Order Information')
    route = pd.read_excel(filePath, sheet_name='Route Information')
    order.loc[order['Journey Type'] == 'Domestic', 'Tax Percentage'] = 0
    route['Cost'] = route[route.columns[7:12]].sum(axis=1)
    route['Time'] = np.ceil(route[route.columns[14:18]].sum(axis=1) / 24)
    route = route[list(route.columns[0:4]) + ['Fixed Freight Cost', 'Time',
                                              'Cost', 'Warehouse Cost', 'Travel Mode', 'Transit Duty'] + list(
        route.columns[-9:-2])]
    route = pd.melt(route, id_vars=route.columns[0:10], value_vars=route.columns[-7:], 
                    var_name='Weekday', value_name='Feasibility')
    route['Weekday'] = route['Weekday'].replace({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
                                                 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})
    return order, route


if __name__ == '__main__':
    order, route = transform("model data.xlsx")
    m = MMT(framework='CVXPY')   # ✅ use CVXPY only
    m.set_param(route, order)
    m.build_model()
    m.solve_model()
    txt = m.txt_solution(route, order)
    with open("Solution.txt", "w") as text_file:
        text_file.write(txt)
