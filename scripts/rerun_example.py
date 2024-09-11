import rerun as rr

# init 3D logging
rr.is_enabled = True
rr.init('example')
rr.serve()

rr.log('points', rr.Points3D([[0, 0, 0]]))
