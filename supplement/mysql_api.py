import aiomysql
from supplement.file_manager import Manage


class Mysql:
    def __init__(self, pool=None):
        self.manage = Manage()
        self.pool = pool
        self.operation = ['=', '>', '<', '<>', '!=', '>=', '<=']

        if not self.pool:
            self.mysql_user_data = self.manage.wr_setting('r')
            if self.mysql_user_data['response']:
                self.db_config = {
                    'host': 'localhost',
                    'user': self.mysql_user_data['content']['sql_admin'],
                    'password': self.mysql_user_data['content']['sql_password'],
                    'port': self.mysql_user_data['content']['sql_port'],
                    'charset': 'utf8mb4'
                }
            else:
                print('Mysql setting mistake. Please retry')
                exit()
        else:
            self.db_config = None

    def __response(self, condition=1, content=None):
        return {'response': condition, 'content': content}

    async def find(self, table, aim='*', order_by=None, order_way=None, relative='', attribute=None, result=None,internal=None, limit=None):
        if isinstance(aim, str):
            sql = 'select ' + aim + f' from {table} '
        elif isinstance(aim, (tuple, list)):
            sql = 'select ' + ' '.join(aim) + f' from {table} '
        else:
            return self.__response(0, 'The aim setting defeat')

        if not internal:
            if isinstance(attribute, (list, tuple)):
                internal = ['='] * len(attribute)
            else:
                internal = '='
        else:
            for i in internal:
                if i not in self.operation:
                    return self.__response(0, f"The find function grammar have mistake,only use {self.operation}")

        k = 0
        if (isinstance(relative, (list, tuple))) and (isinstance(attribute, (list, tuple))) and (
        isinstance(result, (list, tuple))):
            if (len(relative) + 1 == len(attribute)) and (len(attribute) == len(result)) and (
                    len(result) == len(internal)):
                sql = sql + f"where {attribute[0]} {internal[0]} '{result[0]}' "
                for index in range(1, len(attribute)):
                    sql = sql + relative[k] + f" {attribute[index]} {internal[index]} '{result[index]}' "
                    k += 1
            else:
                return self.__response(0, 'This find function is defeat')
        elif (isinstance(relative, str)) and (isinstance(attribute, str)) and (isinstance(result, str)):
            sql = sql + f"where {attribute} {internal} '{result}'"

        if order_by:
            if not order_way:
                order_way = [None] * len(order_by)
            poor = len(order_by) - len(order_way)
            if poor < 0:
                return self.__response(0, 'This function mistake of order_way')
            elif poor > 0:
                for i in range(poor):
                    order_way.append(None)
            sql = sql + 'order by '
            for i in range(len(order_by)):
                if order_way[i] != 'DESC':
                    sql = sql + f'{order_by[i]} ASC'
                else:
                    sql = sql + f'{order_by[i]} {order_way[i]} '
                if i != len(order_by) - 1:
                    sql += ','

        if limit and (isinstance(limit, int)) and (limit > 0):
            sql = sql + f' limit {limit}'
        sql += ';'

        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(sql)
                    data = await cursor.fetchall()
            if hasattr(self, 'mysql_user_data') and self.mysql_user_data and self.mysql_user_data.get('content', {}).get('show_in'):
                print('find code is:', sql)
            return self.__response(content=data)
        except Exception as e:
            return self.__response(0, f"Find error: {str(e)}")

    async def insert(self, table, attribute, values, ret_bool=False):
        if (isinstance(attribute, str)) and (isinstance(values, str)):
            sql = f"insert into {table} ({attribute}) values('{values}')"
        elif isinstance(attribute, (list, tuple)) and isinstance(values, (list, tuple)):
            if len(attribute) != len(values):
                return self.__response(0, 'Insert attribute and values length sql function error!')
            sql = f"insert into {table} ("
            sql = sql + ",".join(attribute) + ") values('"
            sql = sql + "','".join(values) + "')"
        else:
            return self.__response(0, 'Insert attribute and values setting error')
        if hasattr(self, 'mysql_user_data') and self.mysql_user_data and self.mysql_user_data.get('content', {}).get('show_in'):
            print('insert code is:', sql)

        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(sql)
                    await conn.commit()
            if not ret_bool:
                return self.__response(1, 'Insert Success')
            temp = len(values) - 1 if isinstance(values, (tuple, list)) else 0
            return await self.find(table, relative=['and'] * temp, attribute=attribute, result=values,
                                   order_by=attribute)
        except Exception as e:
            return self.__response(0, f"Insert error: {str(e)}")

    async def update(self, table, attribute, values, aim, new_values, ret_bool=False, relative=None, internal=None):
        if (not relative) and (type(attribute) in [tuple, list]):
            relative = ['AND'] * (len(attribute) - 1)
            # 不再将长度为1的attribute转换为空字符串
        elif (not relative) and (type(attribute) is str):
            relative = ''
        elif relative and (type(attribute) in [tuple, list]):
            if type(relative) is str:
                return self.__response(0, 'Relative defeat')
            elif isinstance(relative, list):
                while len(relative) < len(attribute) - 1:
                    relative.append('AND')
                if len(relative) >= len(attribute):
                    return self.__response(0, 'The length lack or superfluous of between relative and attribute')
            elif (isinstance(relative, tuple)) and (len(relative) != len(attribute)):
                return self.__response(0, 'Relative mistake')
            else:
                return self.__response(0, 'Relative parameter mistake.just only use tuple & list & str')
        elif relative and (isinstance(attribute, str)):
            relative = ''
        else:
            return self.__response(0, 'MySql api code error')

        if (not internal) and (type(attribute) in [tuple, list]):
            internal = ['='] * len(attribute)
        elif (not internal) and (isinstance(attribute, str)):
            internal = '='
        elif internal and (type(attribute) in [tuple, list]):
            if isinstance(internal, str):
                return self.__response(0, 'Internal defeat')
            elif isinstance(internal, list):
                while len(internal) < len(attribute):
                    internal.append('=')
                if len(internal) != len(attribute):
                    return self.__response(0, 'The length lack or superfluous of between internal and attribute')
                for i in internal:
                    if i not in self.operation:
                        return self.__response(0, 'Unknown symbol')
            elif (isinstance(internal, tuple)) and (len(internal) != len(attribute)):
                return self.__response(0, 'Internal mistake')
        elif internal and (isinstance(attribute, str)) and (internal not in self.operation):
            internal = '='

        for i in range(len(internal)) if isinstance(internal, (list, tuple)) else []:
            if internal[i] not in self.operation:
                internal[i] = '='

        sql = f'update {table} set'
        if (type(aim) in [tuple, list]) and (type(new_values) in [tuple, list]):
            # 只有当aim和new_values长度相等时才继续
            if len(aim) != len(new_values):
                return self.__response(0, 'Length difference between aim and new_values')
            # 构建set子句
            for i in range(len(aim)):
                sql = sql + f" {aim[i]}='{new_values[i]}'"
                if i != len(aim) - 1:
                    sql += ','
        elif (type(aim) == str) and (type(new_values) in [int, float, str]):
            sql = sql + f" {aim}='{new_values}'"
        else:
            return self.__response(0, 'This function setting defeat')

        k = 0
        if (type(attribute) in [list, tuple]) and (type(values) in [list, tuple]):
            sql = sql + f" where {attribute[0]} {internal[0]} '{values[0]}'"
            for i in range(1, len(attribute)):
                sql = sql + f" {relative[k]} {attribute[i]} {internal[i]} '{values[i]}'"
                k += 1
        elif (type(attribute) == str) and (type(values) in [int, float, str]):
            sql = sql + f" where {attribute} {internal} '{values}'"
        else:
            return self.__response(0, 'This function where defeat')
        sql += ';'
        if hasattr(self, 'mysql_user_data') and self.mysql_user_data and self.mysql_user_data.get('content', {}).get('show_in'):
            print('update code is:', sql)

        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(sql)
                    await conn.commit()
            if not ret_bool:
                return self.__response(1, 'Update Success')

            res_list = []
            res_values = []
            if (isinstance(attribute, str)) and (type(aim) in [list, tuple]):
                if attribute in aim:
                    for i in range(len(aim)):
                        res_list.append(aim[i])
                        if aim[i] == attribute:
                            res_values.append(values)
                        else:
                            res_values.append(new_values[i])
                else:
                    for i in range(len(aim)):
                        res_list.append(aim[i])
                        res_values.append(new_values[i])
                    res_list.append(attribute)
                    res_values.append(values)
            elif (isinstance(attribute, str)) and (isinstance(aim, str)):
                if attribute == aim:
                    res_list.append(aim)
                    res_values.append(new_values)
                else:
                    res_list.append(attribute)
                    res_list.append(aim)
                    res_values.append(values)
                    res_values.append(new_values)
            elif (type(attribute) in [list, tuple]) and (isinstance(aim, str)):
                for i in range(len(attribute)):
                    res_list.append(attribute[i])
                    if attribute[i] == aim:
                        res_values.append(new_values)
                    else:
                        res_values.append(values[i])
            elif (type(attribute) in [list, tuple]) and (type(aim) in [list, tuple]):
                for i in range(len(attribute)):
                    res_list.append(attribute[i])
                    res_values.append(values[i])
                for i in range(len(aim)):
                    if aim[i] in res_list:
                        res_values[res_list.index(aim[i])] = new_values[i]
                    else:
                        res_list.append(aim[i])
                        res_values.append(new_values[i])
            else:
                return self.__response(0, 'Don`t use find')

            new_relative = ['AND'] * (len(res_list) - 1) if len(res_list) - 1 > 0 else None
            return await self.find(table, relative=new_relative, attribute=res_list, result=res_values)
        except Exception as e:
            return self.__response(0, f"Update error: {str(e)}")

    async def delete(self,table,attribute,result,relative=None):
        sql = f'DELETE from {table}'
        if isinstance(attribute,str) and isinstance(result,str):
            sql += f'where {attribute}="{result}"'
        elif isinstance(attribute,(tuple,list)) and isinstance(result,(list,tuple)):
            if len(attribute) != len(result):
                return self.__response(0,'The length is not equal')
            sql += ' where '
            if relative is None:
                relative = ' AND '
            for i in range(len(attribute)):
                sql += f'{attribute[i]}="{result[i]}"'
                if i != len(attribute) - 1:
                    sql += relative
            print('find code is:',sql)
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(sql)
                    await conn.commit()
        except Exception as e:
            return self.__response(0,f"we find a mistake: {e}")
        pass
    
    async def execute(self, sql):
        """直接执行SQL语句"""
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(sql)
                    await conn.commit()
            return self.__response(1, 'Execute Success')
        except Exception as e:
            return self.__response(0, f"Execute error: {str(e)}")