"""Replace only the detector asset with the SRAM-sized 128x224 version."""
import sys
import direct_upload
import load_assets
load_assets.ASSETS = [('small_detector_128x224/text_detector_u55_256.pte', 0xA2400000, 667232)]
if '--elf' not in sys.argv:
    sys.argv.extend(['--elf',str(load_assets.ARTIFACTS / 'flashed_live.elf')])
if '--via-staging' not in sys.argv:
    sys.argv.append('--via-staging')
if '--no-enable' not in sys.argv:
    sys.argv.append('--no-enable')
direct_upload.main()
