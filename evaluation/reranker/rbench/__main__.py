from rbench.cli import main
from rbench.credits import OutOfCredits

try:
    main()
except OutOfCredits as err:  # embedding ran out mid-partition: that partition is encoded again on the rerun
    raise SystemExit(str(err)) from None
