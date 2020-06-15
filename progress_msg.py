import time
import datetime

class ProgressMsg():
    def __init__(self, max):
        self.max = max
        self.start_time = time.time()
        self.progress_time = self.start_time

    def start(self):
        self.start_time = time.time()
        self.progress_time = self.start_time

    def print_prog_msg(self, current):
        if time.time() - self.progress_time < 1:
            return
        self.progress_time = time.time()

        if len(self.max) != len(current):
            raise Exception('current should have same length with max variable.')

        for i in range(len(self.max)):
            if current[i] > self.max[i]:
                raise Exception('current value should be less than max value.')

        pg_per = 0
        for i in reversed(range(len(self.max))):
            pg_per += current[i]
            pg_per /= self.max[i]
        pg_per *= 100

        if pg_per != 0:
            elapsed = time.time() - self.start_time
            total = 100*elapsed/pg_per
            remain = total - elapsed
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
            remain_str = str(datetime.timedelta(seconds=int(remain)))
            total_str = str(datetime.timedelta(seconds=int(total)))
            txt = '>>> progress : %.2f%%, elapsed: %s, remaining: %s, total: %s \t\t\t\t\t' % (pg_per, elapsed_str, remain_str, total_str)
        else:
            elapsed = time.time() - self.start_time
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
            txt = '>>> progress : %.2f%%, elapsed: %s, remaining: %s, total: %s \t\t\t\t\t' % (pg_per, elapsed_str, 'INF', 'INF')

        
        print(txt, end='\r')

        return txt.replace('\t', '')

    def print_finish_msg(self):
        total = time.time() - self.start_time
        total_str = str(datetime.timedelta(seconds=int(total)))
        txt = 'Finish >>> (total elapsed time : %s) \t\t\t\t\t' % total_str
        print(txt)
        return txt.replace('\t', '')

if __name__ == '__main__':
    pp = ProgressMsg((100,10))
    pp.start()
    
    print(pp.__class__.__name__)

    for i in range(20):
        pp.print_prog_msg((0, i))
        time.sleep(1)
        


