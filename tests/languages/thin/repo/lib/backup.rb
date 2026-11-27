# Rotating backups: keep the newest few copies of a folder.
module Backup
  KEEP = 7

  class Rotator
    def initialize(dir)
      @dir = dir
    end

    def prune
      copies = Dir.glob(File.join(@dir, "*.tar.gz")).sort
      (copies[0...-KEEP] || []).each { |f| File.delete(f) }
    end
  end

  def self.stamp(time)
    time.strftime("%Y%m%d-%H%M%S")
  end
end
