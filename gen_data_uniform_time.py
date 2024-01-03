import os
import numpy as np
import random


def generate(case_name, path, change_rate=0.2, bound=0.1):
    job_input = {}
    orders_of_job = {}
    file = path + case_name + ".fjs"
    with open(file, 'r') as f:
        user_line = f.readline()
        user_line = str(user_line).replace('\n', ' ')
        user_line = str(user_line).replace('\t', ' ')
        data = user_line.split(' ')
        while data.__contains__(""):
            data.remove("")
        m_n = list(map(int, data))

        for job in range(int(m_n[0])):
            user_line = f.readline()
            user_line = str(user_line).replace('\n', ' ')
            user_line = str(user_line).replace('\t', ' ')
            data = user_line.split(' ')
            while data.__contains__(""):
                data.remove("")
            line_data = list(map(int, data))

            num_of_orders = line_data[0]
            orders_of_job[job] = num_of_orders
            k = 1
            for i in range(num_of_orders):
                num_of_machines = line_data[k]
                machines = []
                processing_time = []
                for j in range(num_of_machines):
                    machines.append(line_data[j * 2 + k + 1])
                    processing_time.append(line_data[j * 2 + k + 2])
                job_input[job, i * 2] = machines
                job_input[job, i * 2 + 1] = processing_time
                k = k + 2 * num_of_machines + 1
    f.close()
    job_num = m_n[0]
    machine_num = m_n[1]
    max_op_len = 0
    # find maximum operation length of all jobs
    for j in range(job_num):
        machines = orders_of_job[j]
        for i in range(machines):
            ops = job_input[j, i * 2 + 1]
            if max_op_len < max(ops):
                max_op_len = max(ops)
    for j in range(job_num):
        for i in range(orders_of_job[j]):
            if random.random() < change_rate:
                process_time_set = job_input[j, i * 2 + 1]
                if bound <= 1:
                    new_process_time = round(process_time_set[0]*random.uniform(1-bound, 1+bound))
                else:
                    new_process_time = round(random.uniform(1, max_op_len))
                process_time_set[0] = new_process_time
                job_input[j, i * 2 + 1] = process_time_set
    output_path = "base_instance_uncertain_time/"
    new_instance = output_path + case_name + "_" + str(int(change_rate*100)) + "_" + str(int(100*bound)) + ".fjs"
    file = open(new_instance, mode='w')
    file.write(str(job_num)+'\t')
    file.write(str(machine_num))
    file.write('\n')
    for j in range(job_num):
        jobi = [str(orders_of_job[j]), '\t']
        for i in range(orders_of_job[j]):
            time_set = job_input[j, i * 2 + 1]
            machine_set = job_input[j, i * 2]
            jobi.append(str(len(time_set)))
            jobi.append(" ")
            for k in range(len(time_set)):
                jobi.append(str(machine_set[k]))
                jobi.append(' ')
                jobi.append(str(time_set[k]))
                jobi.append(' ')
        jobi.append('\n')
        file.writelines(jobi)
    file.close()


if __name__ == '__main__':
    file_path = 'base_instance/'
    for file_name in os.listdir(file_path):
        print(file_name + "========================")
        title = file_name.split('.')[0]
        for m in range(4):
            for n in range(3):
                generate(title, file_path, change_rate=(m+1)*0.25, bound=(n+1)*0.25)



