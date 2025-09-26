BEGIN {
    sent=0; received=0;
}

# Sent = enqueue TCP packet into the link
$1 == "+" && $5 == "tcp" { sent++ }

# Received = TCP packet received at destination
$1 == "r" && $5 == "tcp" { received++ }

END {
    if (sent == 0) {
        print "No TCP packets were sent!"
    } else {
        pdr = (received/sent)*100
        plr = ((sent-received)/sent)*100
        printf("Packets Sent = %d\n", sent)
        printf("Packets Received = %d\n", received)
        printf("PDR = %.2f%%\n", pdr)
        printf("PLR = %.2f%%\n", plr)
    }
}
