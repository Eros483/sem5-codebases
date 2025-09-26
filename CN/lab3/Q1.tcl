# Define simulator
set ns [new Simulator]

# Trace file
set tf [open Q2.tr w]
$ns trace-all $tf

# Finish procedure
proc finish {} {
    global ns tf
    $ns flush-trace
    close $tf
    exec awk -f Q3.awk Q2.tr &
    exit 0
}

# Create nodes
set n0 [$ns node]
set n1 [$ns node]

# Full duplex link -- Modifiable Parameters
$ns duplex-link $n0 $n1 0.5Mb 10ms DropTail
$ns queue-limit $n0 $n1 10

# TCP agent at n0
set tcp [new Agent/TCP]
$tcp set class_ 1
$ns attach-agent $n0 $tcp

# TCP sink at n1
set sink [new Agent/TCPSink]
$ns attach-agent $n1 $sink

# Connect TCP and sink
$ns connect $tcp $sink

# FTP application
set ftp [new Application/FTP]
$ftp attach-agent $tcp

# Generate traffic
$ns at 0.1 "$ftp start"
$ns at 4.0 "$ftp send 50000"   ;# send 50,000 bytes
$ns at 9.9 "$ftp stop"

# End simulation
$ns at 10.0 "finish"

# Run simulation
$ns run
