"""Recover PSRAM/peripherals using the shared board's reset pin."""
import fcntl,json
from restore_original import connection,ROOT
with open('/tmp/alif-e8-1219599493.lock','a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 link=connection()
 try:
  link.set_reset_strategy(2)
  try:
   result={'reset_result':link.reset(ms=100,halt=False)}
  except Exception as exc:
   result={'reset_exception':str(exc),'note':'Reset pin may drop debug; reconnect in a fresh process.'}
  print(json.dumps(result),flush=True)
 finally:link.close()
