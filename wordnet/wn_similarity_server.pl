#!/usr/bin/perl -w

use strict;
use WordNet::QueryData;
use WordNet::Similarity::hso;
use HTTP::Daemon;
use HTTP::Status;

my$wn=WordNet::QueryData->new;
my$measure=WordNet::Similarity::hso->new($wn);

my $daemon=HTTP::Daemon->new(
    LocalAddr => '127.0.0.1',
    LocalPort => 10137
    );
die "Can't create: $!" unless($daemon);
print "Please contact me at: <URL:", $daemon->url, ">\n";
while (1){
    my $conn=$daemon->accept;
    while (1){
	my $req = $conn->get_request;
	last unless($req);
	if($req->method eq 'GET' and $req->url->path =~ m-^/sim/[^/]+/.+-) {
	    my($term1,$term2) = $req->url->path =~ m-^/sim/([^/]+)/(.+)-;
	    my$sim=&getMaxSim($term1,$term2,$measure,$wn);
	    print"$term1 $term2 $sim\n";
	    $conn->send_response(200, "Success", undef, "$sim");
	} else {
              $conn->send_error(RC_FORBIDDEN)
	}
    }
    $conn->close;
    undef($conn);
}


sub getSynSets {
    my$term=shift;
    my$wn=shift;
    my@pos=$wn->querySense($term);
    my@valid=();
    foreach my$pos(@pos){
	push@valid,$wn->querySense($pos);
    }
    return @valid;
}

sub getMaxSim {
    my($term1,$term2, $measure, $wn) = @_;
    my@synSets1 = &getSynSets($term1, $wn);
    my@synSets2 = &getSynSets($term2, $wn);
    my$maxSim=-999999;
    foreach my$synSet1(@synSets1){
	foreach my$synSet2(@synSets2){
	    my$sim=$measure->getRelatedness($synSet1,$synSet2);
	    my($error,$errorStr)=$measure->getError();
	    if($error){
		print STDERR "For $synSet1 $synSet2: $errorStr\n";
	    }elsif($sim>$maxSim){
		$maxSim=$sim;
	    }
	}
    }
    return $maxSim;
}
