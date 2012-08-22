#!/usr/bin/perl -w

use strict;
use WordNet::QueryData;

my($vocab_file,$tar_dir_file,$wnsimil_distr,$tmp_dir,$out_dir)=@ARGV;

`mkdir -p $tmp_dir`;

my$wn=WordNet::QueryData->new;
#
# read vocabulary
#
open(V,"$vocab_file") or die "$!";
my@vocab=<V>; chomp @vocab;
close V;

#
# read tar file directory
#
open(D,"$tar_dir_file") or die "$!";
my@dir_file=();
while(<D>){
    chomp;
    next unless(m/\.txt.gz$/);
    s/^.*WordNet-/WordNet-/;
    push@dir_file,$_;
}
close D;

#
# extract relevant files
#
my%extracted_files=map { $_ => [] } @vocab;
open(F,">$tmp_dir/nouns_to_extract.lst")or die"$!";
my%vocab_set=map {$_=>1} @vocab;
foreach(@dir_file){
    my$file=$_;
    s|WordNet-[^/]+\/[^/]+\/||;
    s|\#.*$||;
    if($vocab_set{$_}){
	print F "$file\n";
	push@{$extracted_files{$_}},$file;
    }
}
close F;
open(F,">$tmp_dir/verbs_to_extract.lst")or die"$!";
my@verbs=split(/\n/,`tar -tvf $wnsimil_distr/WordNet-verb-verb-path-pairs.tar`);
foreach(@verbs){
    next unless(m/\.txt.gz$/);
    s/^.*WordNet-/WordNet-/;
    my$file=$_;
    s|WordNet-[^/]+\/[^/]+\/||;
    s|\#.*$||;
    if($vocab_set{$_}){
	print F "$file\n";
	push@{$extracted_files{$_}},$file;
    }
}
close F;

`cd $tmp_dir; tar --files-from $tmp_dir/nouns_to_extract.lst -xf $wnsimil_distr/WordNet-noun-noun-path-pairs.tar`;
`cd $tmp_dir; tar --files-from $tmp_dir/verbs_to_extract.lst -xf $wnsimil_distr/WordNet-verb-verb-path-pairs.tar`;

#
# merge files leaving behind sense and POS information
#
`mkdir -p $out_dir`;
foreach my$vocab(@vocab){
    my%scores=();
    foreach my$file(@{$extracted_files{$vocab}}){
	open(F,"zcat $tmp_dir/$file|")or die"$!";
	my$first=1;
	while(<F>){
	    if($first){
		$first=0;
		next;
	    }
	    chomp;
	    my($score,$sense)=m/(.*) (.*)/;
	    my@syns=$wn->querySense($sense,"syns");
	    foreach my $syn(@syns){
		my($word)=$syn=~m/(.*)\#.*\#.*/;
		$scores{$word} = $score if(!defined($scores{$word}) or
					   $scores{$word} < $score);
	    }
	}
	close F;
    }
    open(O,">$out_dir/$vocab.scores") or die"$!";
    foreach my$word(keys %scores){
	print O $scores{$word} . " " . $word . "\n";
    }
    close O;
}
