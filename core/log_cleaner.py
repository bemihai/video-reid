import os
import re
import glob
from datetime import date, datetime
import numpy as np


# ------------------------------------------------------------------------------------------------

# Note: to redirect both stdout and stderr of a python process (in Windows) into the
# same log file whose name depends on the current date, the following command can be used:
# python script.py 1 >> logs-%date:~10,4%-%date:~4,2%-%date:~7,2%.log 2>>&1
#
# The file name format for the above command is "logs-[year]-[month]-[day].log".
# This format is set as the default one for the "clean" method below.


# ------------------------------------------------------------------------------------------------

def clean(root_dir, log_file_format='.*-{Y}-{M}-{D}\..*', max_age_days=1):
    """
    :param root_dir: root directory containing log files.
    :param log_file_format: regular expression containing tokens for day, month, year.
                            See also date_regex_str parameter in the constructor of DateRegex.
    :param max_age_days: maximum number of days that need to have passed in order for the
                         respective log to be deleted.
    """

    files = glob.glob(root_dir)
    print('current log files:', files)
    now = datetime.now().date()
    dr = DateRegex(log_file_format)

    for file in files:
        date = dr.parse(file)
        if date is None:
            print('unexpected file name:', file)
            continue

        age = (now - date).days
        if age > max_age_days:
            print('deleting old log file:', file)
            os.remove(file)


# ------------------------------------------------------------------------------------------------

class DateRegex:
    def __init__(self, date_regex_str):
        """
        :param date_regex_str: regex style string with some specific tokens for
                               specifying year {Y} (expects a 4 digit integer),
                               month {M} (integer), and day {D} (integer).
        """

        self._dmy_idx, self._exp = \
            self._compile_regex(date_regex_str)


# ------------------------------------------------------------------------------------------------

    def _compile_regex(self, date_regex_str):
        d_pos = date_regex_str.find('{D}')
        m_pos = date_regex_str.find('{M}')
        y_pos = date_regex_str.find('{Y}')

        # all the date tokens ({D}, {M} and {Y}) must exist
        if d_pos == -1 or m_pos == -1 or y_pos == -1:
            assert False

        idxs = np.argsort([d_pos, m_pos, y_pos])

        date_regex_str = date_regex_str.replace('{D}', '(\d+)')
        date_regex_str = date_regex_str.replace('{M}', '(\d+)')
        date_regex_str = date_regex_str.replace('{Y}', '(\d\d\d\d)')
        exp = re.compile(date_regex_str)
        return idxs, exp


# ------------------------------------------------------------------------------------------------

    def parse(self, str):
        """
        Parse the input string according to this date regex.
        Returns the date object matched in the string, or None if the string
        does not match the expected pattern.
        """
        values = self._exp.findall(str)
        if values is None or len(values) == 0:
            return None

        values = values[0]
        assert(len(values) == 3)

        day = int(values[self._dmy_idx[0]])
        month = int(values[self._dmy_idx[1]])
        year = int(values[self._dmy_idx[2]])

        return date(year, month, day)


# ------------------------------------------------------------------------------------------------

def _test():
    dr = DateRegex('.*-{D}-{M}-{Y}\.log')
    date = dr.parse('log-12-11-2018.log')
    print('parsed:', date)
    now = datetime.now().date()
    print('days:', (now - date).days)

    clean(r'../../logs/*',
          log_file_format='.*-{Y}-{M}-{D}\..*', max_age_days=1)


# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    _test()


# ------------------------------------------------------------------------------------------------
