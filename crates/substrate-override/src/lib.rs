//! # `substrate-override`
//!
//! Clean-room Rust implementation of the **3-Layer Virtual Block & In-Flight Binary Mutation Engine**
//! reverse-engineered from `ventoy/Ventoy`:
//! - `EDK2/edk2_mod/.../Ventoy/Ventoy.h` (`ventoy_os_param`, `ventoy_chain_head`, `ventoy_img_chunk`,
//!   `ventoy_override_chunk`, `ventoy_virt_chunk`)
//! - `EDK2/edk2_mod/.../Ventoy/VentoyProtocol.c` (`ventoy_block_io_read`, `ventoy_read_iso_sector`,
//!   in-flight override patching, and virtual sector remapping)
//! - `GRUB2/MOD_SRC/.../ventoy/ventoy_linux.c` (uncompressed `newc` CPIO archive synthesis & prepending)
//! - `VtoyTool/vtoydm.c` (Linux kernel Device-Mapper `linear` target table generator)

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Exact 16-byte magic GUID from `Ventoy.h`:
/// `{ 0x77772020, 0x2e77, 0x6576, { 0x6e, 0x74, 0x6f, 0x79, 0x2e, 0x6e, 0x65, 0x74 } }`
/// which encodes the ASCII string `"  www.ventoy.net"` in little-endian layout.
pub const VENTOY_GUID_BYTES: [u8; 16] = *b"  www.ventoy.net";

/// Logical sector size of an ISO9660 / virtual image chunk (2048 bytes).
pub const VIRT_SECTOR_SIZE: u64 = 2048;

/// Physical disk sector size (512 bytes).
pub const PHYS_SECTOR_SIZE: u64 = 512;

#[derive(Debug, Error)]
pub enum SubstrateError {
    #[error("Invalid OS param magic GUID")]
    InvalidGuid,
    #[error("Invalid OS param checksum: sum={0:#04x} (expected 0x00)")]
    InvalidChecksum(u8),
    #[error("Read out of virtual bounds: offset={offset}, len={len}, virt_size={virt_size}")]
    OutOfBounds { offset: u64, len: usize, virt_size: u64 },
    #[error("Unmapped virtual sector {0}")]
    UnmappedSector(u64),
}

/// Cross-stage 512-byte survival parameter block (`ventoy_os_param` in `Ventoy.h:76-111`).
/// Passed across pre-boot / runtime boundaries via UEFI variables, `EfiRuntimeServicesData`,
/// low memory (`0x80000`), PE payload concatenation, or CPIO header injection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OsParam {
    pub guid: [u8; 16],
    pub chksum: u8,
    pub disk_guid: [u8; 16],
    pub disk_size: u64,
    pub disk_part_id: u16,
    pub disk_part_type: u16,
    pub img_path: String,
    pub img_size: u64,
    pub img_location_addr: u64,
    pub img_location_len: u32,
    pub reserved: [u8; 32],
    pub disk_signature: [u8; 4],
}

impl OsParam {
    pub fn new(disk_guid: [u8; 16], disk_signature: [u8; 4], img_path: &str, img_size: u64) -> Self {
        let mut param = Self {
            guid: VENTOY_GUID_BYTES,
            chksum: 0,
            disk_guid,
            disk_size: img_size,
            disk_part_id: 1,
            disk_part_type: 0,
            img_path: img_path.to_string(),
            img_size,
            img_location_addr: 0,
            img_location_len: 0,
            reserved: [0u8; 32],
            disk_signature,
        };
        let raw = param.to_bytes();
        param.chksum = raw[16];
        param
    }

    /// Serializes `OsParam` into the exact 512-byte wire layout (`Ventoy.h:76-111`)
    /// and computes the modular 1-byte checksum (`0 - sum(bytes) mod 256`).
    pub fn to_bytes(&self) -> [u8; 512] {
        let mut buf = [0u8; 512];
        buf[0..16].copy_from_slice(&self.guid);
        // buf[16] is chksum, filled last
        buf[17..33].copy_from_slice(&self.disk_guid);
        buf[33..41].copy_from_slice(&self.disk_size.to_le_bytes());
        buf[41..43].copy_from_slice(&self.disk_part_id.to_le_bytes());
        buf[43..45].copy_from_slice(&self.disk_part_type.to_le_bytes());

        let path_bytes = self.img_path.as_bytes();
        let copy_len = path_bytes.len().min(383);
        buf[45..45 + copy_len].copy_from_slice(&path_bytes[..copy_len]);

        buf[429..437].copy_from_slice(&self.img_size.to_le_bytes());
        buf[437..445].copy_from_slice(&self.img_location_addr.to_le_bytes());
        buf[445..449].copy_from_slice(&self.img_location_len.to_le_bytes());
        buf[449..481].copy_from_slice(&self.reserved);
        buf[481..485].copy_from_slice(&self.disk_signature);

        let sum: u8 = buf.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        buf[16] = 0u8.wrapping_sub(sum);
        buf
    }

    /// Validates and deserializes a 512-byte `ventoy_os_param` block (`Ventoy.c:CheckOsParam`).
    pub fn from_bytes(buf: &[u8; 512]) -> Result<Self, SubstrateError> {
        if buf[0..16] != VENTOY_GUID_BYTES {
            return Err(SubstrateError::InvalidGuid);
        }
        let total_sum: u8 = buf.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        if total_sum != 0 {
            return Err(SubstrateError::InvalidChecksum(total_sum));
        }

        let mut guid = [0u8; 16];
        guid.copy_from_slice(&buf[0..16]);
        let chksum = buf[16];
        let mut disk_guid = [0u8; 16];
        disk_guid.copy_from_slice(&buf[17..33]);
        let disk_size = u64::from_le_bytes(buf[33..41].try_into().unwrap());
        let disk_part_id = u16::from_le_bytes(buf[41..43].try_into().unwrap());
        let disk_part_type = u16::from_le_bytes(buf[43..45].try_into().unwrap());

        let path_slice = &buf[45..429];
        let nul_pos = path_slice.iter().position(|&b| b == 0).unwrap_or(path_slice.len());
        let img_path = String::from_utf8_lossy(&path_slice[..nul_pos]).into_owned();

        let img_size = u64::from_le_bytes(buf[429..437].try_into().unwrap());
        let img_location_addr = u64::from_le_bytes(buf[437..445].try_into().unwrap());
        let img_location_len = u32::from_le_bytes(buf[445..449].try_into().unwrap());
        let mut reserved = [0u8; 32];
        reserved.copy_from_slice(&buf[449..481]);
        let mut disk_signature = [0u8; 4];
        disk_signature.copy_from_slice(&buf[481..485]);

        Ok(Self {
            guid,
            chksum,
            disk_guid,
            disk_size,
            disk_part_id,
            disk_part_type,
            img_path,
            img_size,
            img_location_addr,
            img_location_len,
            reserved,
            disk_signature,
        })
    }

    /// Scans an arbitrary binary image on 16-byte boundaries (`vtoyjump.c:2834-2845`)
    /// to recover an embedded `OsParam`.
    pub fn scan_aligned_16(haystack: &[u8]) -> Option<(usize, Self)> {
        let mut offset = 0;
        while offset + 512 <= haystack.len() {
            if haystack[offset..offset + 16] == VENTOY_GUID_BYTES {
                let block: &[u8; 512] = haystack[offset..offset + 512].try_into().ok()?;
                if let Ok(param) = Self::from_bytes(block) {
                    return Some((offset, param));
                }
            }
            offset += 16;
        }
        None
    }
}

/// Layer 1: Physical Fragmentation Mapping (`ventoy_img_chunk` in `Ventoy.h:117-123`).
/// Maps contiguous 2048-byte virtual image sectors `[img_start_sector..=img_end_sector]`
/// to 512-byte physical disk LBAs `[disk_start_sector..=disk_end_sector]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImgChunk {
    pub img_start_sector: u32,
    pub img_end_sector: u32,
    pub disk_start_sector: u64,
    pub disk_end_sector: u64,
}

/// Layer 2: In-Flight Byte-Range Override (`ventoy_override_chunk` in `Ventoy.h:127-132`).
/// Patches arbitrary byte ranges `[img_offset .. img_offset + override_data.len()]` in RAM
/// as sectors are read, without mutating the backing storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OverrideChunk {
    pub img_offset: u64,
    pub override_data: Vec<u8>,
}

/// Layer 3: Appended Virtual Sector Descriptor (`ventoy_virt_chunk` in `Ventoy.h:135-147`).
/// Serves sectors beyond `real_img_size_in_bytes` from either:
/// 1. An in-RAM buffer (`mem_sector_start..mem_sector_end`), and/or
/// 2. A remapped tail (`remap_sector_start..remap_sector_end`) redirected back to `org_sector_start`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VirtChunk {
    pub mem_sector_start: u32,
    pub mem_sector_end: u32,
    pub mem_payload: Vec<u8>,
    pub remap_sector_start: u32,
    pub remap_sector_end: u32,
    pub org_sector_start: u32,
}

/// Complete 3-Layer Virtual Substrate (`ventoy_chain_head` + `VentoyProtocol.c:204-598`).
/// Wraps an immutable backing store and applies:
/// - Fragment translation (`img_chunks`)
/// - In-place byte overrides (`overrides`)
/// - Appended virtual/remapped sectors (`virt_chunks`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualSubstrate {
    pub os_param: OsParam,
    pub real_img_size_in_bytes: u64,
    pub virt_img_size_in_bytes: u64,
    pub img_chunks: Vec<ImgChunk>,
    pub overrides: Vec<OverrideChunk>,
    pub virt_chunks: Vec<VirtChunk>,
    backing_storage: Vec<u8>,
}

impl VirtualSubstrate {
    /// Creates a new `VirtualSubstrate` over an immutable backing image buffer.
    pub fn new(backing_storage: Vec<u8>, img_path: &str) -> Self {
        let real_len = backing_storage.len() as u64;
        let aligned_sectors = real_len.div_ceil(VIRT_SECTOR_SIZE) as u32;
        let disk_sectors = real_len.div_ceil(PHYS_SECTOR_SIZE);

        let chunk = ImgChunk {
            img_start_sector: 0,
            img_end_sector: aligned_sectors.saturating_sub(1),
            disk_start_sector: 0,
            disk_end_sector: disk_sectors.saturating_sub(1),
        };

        let os_param = OsParam::new(*b"SINGULARITY-DISK", *b"SNG1", img_path, real_len);

        Self {
            os_param,
            real_img_size_in_bytes: real_len,
            virt_img_size_in_bytes: (aligned_sectors as u64) * VIRT_SECTOR_SIZE,
            img_chunks: vec![chunk],
            overrides: Vec::new(),
            virt_chunks: Vec::new(),
            backing_storage,
        }
    }

    /// Forks the substrate into a lightweight child clone sharing the base image
    /// with a cleared or inherited override table (replaces `GitBasedOrganism.build_repo()`).
    pub fn fork_with_overrides(&self, additional_overrides: Vec<OverrideChunk>) -> Self {
        let mut child = self.clone();
        child.overrides.extend(additional_overrides);
        child
    }

    /// Anti-Grey-Goo Compaction: If the override delta log exceeds `max_overrides`,
    /// flattens all active `OverrideChunk` byte patches into `backing_storage` and clears `overrides`.
    pub fn compact_overrides(&mut self, max_overrides: usize) {
        if self.overrides.len() <= max_overrides {
            return;
        }
        for ovr in &self.overrides {
            let start = ovr.img_offset as usize;
            for (i, &b) in ovr.override_data.iter().enumerate() {
                if start + i < self.backing_storage.len() {
                    self.backing_storage[start + i] = b;
                }
            }
        }
        self.overrides.clear();
    }

    /// Registers an in-flight byte override at `img_offset` (`VentoyProtocol.c:204-250`).
    pub fn add_override(&mut self, img_offset: u64, override_data: Vec<u8>) {
        self.overrides.push(OverrideChunk {
            img_offset,
            override_data,
        });
    }

    /// Appends a virtual chunk past the end of the image (`ventoy_linux.c:784-838`),
    /// optionally remapping tail sectors back to `org_sector_start` + `org_sector_count`.
    pub fn append_virt_chunk(
        &mut self,
        mem_payload: Vec<u8>,
        org_sector_start: u32,
        org_sector_count: u32,
    ) -> u32 {
        let start_sector = (self.virt_img_size_in_bytes / VIRT_SECTOR_SIZE) as u32;
        let mem_sectors = (mem_payload.len() as u64).div_ceil(VIRT_SECTOR_SIZE) as u32;
        let mut padded_payload = mem_payload;
        padded_payload.resize((mem_sectors as usize) * (VIRT_SECTOR_SIZE as usize), 0);

        let mem_sector_end = start_sector + mem_sectors;
        let remap_sector_start = mem_sector_end;
        let remap_sector_end = remap_sector_start + org_sector_count;

        self.virt_chunks.push(VirtChunk {
            mem_sector_start: start_sector,
            mem_sector_end,
            mem_payload: padded_payload,
            remap_sector_start,
            remap_sector_end,
            org_sector_start,
        });

        self.virt_img_size_in_bytes = (remap_sector_end as u64) * VIRT_SECTOR_SIZE;
        start_sector
    }

    /// Reads arbitrary virtual byte ranges applying all 3 layers (`ImgChunk` -> `OverrideChunk` -> `VirtChunk`),
    /// faithfully mirroring `ventoy_block_io_read` (`VentoyProtocol.c:481-598`).
    pub fn read_bytes(&self, offset: u64, len: usize) -> Result<Vec<u8>, SubstrateError> {
        if offset + (len as u64) > self.virt_img_size_in_bytes {
            return Err(SubstrateError::OutOfBounds {
                offset,
                len,
                virt_size: self.virt_img_size_in_bytes,
            });
        }

        let mut out = vec![0u8; len];
        let mut pos = 0usize;

        while pos < len {
            let cur_offset = offset + (pos as u64);
            let sector = (cur_offset / VIRT_SECTOR_SIZE) as u32;
            let sector_off = (cur_offset % VIRT_SECTOR_SIZE) as usize;
            let to_copy = (len - pos).min((VIRT_SECTOR_SIZE as usize) - sector_off);

            let sector_buf = self.read_sector_2048(sector)?;
            out[pos..pos + to_copy].copy_from_slice(&sector_buf[sector_off..sector_off + to_copy]);
            pos += to_copy;
        }

        Ok(out)
    }

    /// Reads a single 2048-byte virtual sector through the 3-layer translation pipeline.
    pub fn read_sector_2048(&self, sector: u32) -> Result<[u8; 2048], SubstrateError> {
        let byte_offset = (sector as u64) * VIRT_SECTOR_SIZE;

        // Layer 1 & 2: Inside real image bounds
        if byte_offset < self.real_img_size_in_bytes {
            let mut buf = self.read_real_sector_raw(sector)?;
            self.apply_overrides(byte_offset, &mut buf);
            return Ok(buf);
        }

        // Layer 3: Inside appended virtual sectors (`VentoyProtocol.c:481-558`)
        for virt in &self.virt_chunks {
            if sector >= virt.mem_sector_start && sector < virt.mem_sector_end {
                let rel_sec = (sector - virt.mem_sector_start) as usize;
                let start = rel_sec * (VIRT_SECTOR_SIZE as usize);
                let mut buf = [0u8; 2048];
                buf.copy_from_slice(&virt.mem_payload[start..start + 2048]);
                return Ok(buf);
            }
            if sector >= virt.remap_sector_start && sector < virt.remap_sector_end {
                let org_sec = virt.org_sector_start + (sector - virt.remap_sector_start);
                let org_byte_offset = (org_sec as u64) * VIRT_SECTOR_SIZE;
                let mut buf = self.read_real_sector_raw(org_sec)?;
                self.apply_overrides(org_byte_offset, &mut buf);
                return Ok(buf);
            }
        }

        Err(SubstrateError::UnmappedSector(sector as u64))
    }

    fn read_real_sector_raw(&self, img_sector: u32) -> Result<[u8; 2048], SubstrateError> {
        for chunk in &self.img_chunks {
            if img_sector >= chunk.img_start_sector && img_sector <= chunk.img_end_sector {
                let sector_delta = (img_sector - chunk.img_start_sector) as u64;
                let phys_byte_start =
                    chunk.disk_start_sector * PHYS_SECTOR_SIZE + sector_delta * VIRT_SECTOR_SIZE;
                let mut buf = [0u8; 2048];
                let start = phys_byte_start as usize;
                if start < self.backing_storage.len() {
                    let avail = (self.backing_storage.len() - start).min(2048);
                    buf[..avail].copy_from_slice(&self.backing_storage[start..start + avail]);
                }
                return Ok(buf);
            }
        }
        Err(SubstrateError::UnmappedSector(img_sector as u64))
    }

    /// Applies overlapping `OverrideChunk` patches (`VentoyProtocol.c:204-250`).
    fn apply_overrides(&self, read_offset: u64, buf: &mut [u8]) {
        let read_end = read_offset + (buf.len() as u64);
        for ovr in &self.overrides {
            let ovr_start = ovr.img_offset;
            let ovr_end = ovr_start + (ovr.override_data.len() as u64);
            if ovr_end <= read_offset || ovr_start >= read_end {
                continue;
            }
            let overlap_start = ovr_start.max(read_offset);
            let overlap_end = ovr_end.min(read_end);
            let buf_idx = (overlap_start - read_offset) as usize;
            let ovr_idx = (overlap_start - ovr_start) as usize;
            let copy_len = (overlap_end - overlap_start) as usize;
            buf[buf_idx..buf_idx + copy_len]
                .copy_from_slice(&ovr.override_data[ovr_idx..ovr_idx + copy_len]);
        }
    }

    /// Generates the exact Linux kernel Device-Mapper `linear` target table
    /// reverse-engineered from `VtoyTool/vtoydm.c:188-233`.
    pub fn generate_dmsetup_linear_table(&self, disk_device: &str) -> String {
        let mut lines = Vec::with_capacity(self.img_chunks.len());
        for chunk in &self.img_chunks {
            let disk_sector = chunk.disk_start_sector;
            let curr_sector = (chunk.img_start_sector as u64) * 4;
            let sector_num = ((chunk.img_end_sector - chunk.img_start_sector + 1) as u64) * 4;
            lines.push(format!(
                "{} {} linear {} {}",
                curr_sector, sector_num, disk_device, disk_sector
            ));
        }
        lines.join("\n")
    }
}

/// Synthesizes a valid uncompressed SVR4 `newc` (`070701`) CPIO archive in memory
/// (reverse-engineered from `ventoy_linux.c:784-885`) so files can be prepended
/// in flight ahead of any compressed Linux `initramfs`.
pub fn build_newc_cpio_archive(files: &[(&str, &[u8], u32)]) -> Vec<u8> {
    let mut archive = Vec::new();
    for (idx, &(name, data, mode)) in files.iter().enumerate() {
        write_newc_entry(&mut archive, (idx + 1) as u32, name, data, mode);
    }
    write_newc_entry(&mut archive, 0, "TRAILER!!!", &[], 0);
    // Align archive to 512-byte boundary as required by Linux init/initramfs.c
    while archive.len() % 512 != 0 {
        archive.push(0);
    }
    archive
}

fn write_newc_entry(out: &mut Vec<u8>, ino: u32, name: &str, data: &[u8], mode: u32) {
    let namesize = name.len() + 1;
    let header = format!(
        "070701{:08X}{:08X}{:08X}{:08X}{:08X}{:08X}{:08X}{:08X}{:08X}{:08X}{:08X}{:08X}{:08X}",
        ino,
        mode,
        0u32, // uid
        0u32, // gid
        1u32, // nlink
        0u32, // mtime
        data.len() as u32,
        0u32, // devmajor
        0u32, // devminor
        0u32, // rdevmajor
        0u32, // rdevminor
        namesize as u32,
        0u32  // check
    );
    debug_assert_eq!(header.len(), 110);
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(name.as_bytes());
    out.push(0); // NUL terminator
    while out.len() % 4 != 0 {
        out.push(0);
    }
    out.extend_from_slice(data);
    while out.len() % 4 != 0 {
        out.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_os_param_roundtrip_and_16b_scanner() {
        let param = OsParam::new(
            *b"0123456789ABCDEF",
            [0xDE, 0xAD, 0xBE, 0xEF],
            "/ISO/ubuntu-24.04.iso",
            4_294_967_296,
        );
        let bytes = param.to_bytes();

        let mut fake_pe = vec![0x90u8; 64]; // 16-byte aligned prefix
        fake_pe.extend_from_slice(&bytes);
        fake_pe.extend_from_slice(b"ORIGINAL_WINPESHL_PAYLOAD");

        let (found_off, decoded) = OsParam::scan_aligned_16(&fake_pe).expect("must find OsParam");
        assert_eq!(found_off, 64);
        assert_eq!(decoded, param);
    }

    #[test]
    fn test_three_layer_substrate_override_and_virt_remap() {
        let mut base = vec![0xAAu8; 4096]; // 2 sectors of 2048B
        base[2048..2056].copy_from_slice(b"ORIGTAIL");

        let mut substrate = VirtualSubstrate::new(base, "/test.iso");
        // Layer 2: Patch bytes 10..18 in sector 0 without touching backing_storage
        substrate.add_override(10, b"MUTATED!".to_vec());

        // Layer 3: Append 2048B synthetic CPIO header + remap sector 1 at the tail
        let virt_start_sec = substrate.append_virt_chunk(b"VENTOY_CPIO_HDR".to_vec(), 1, 1);
        assert_eq!(virt_start_sec, 2);

        let sec0 = substrate.read_bytes_checked(10, 8);
        assert_eq!(&sec0, b"MUTATED!");

        let sec2_virt = substrate.read_bytes_checked(2 * 2048, 15);
        assert_eq!(&sec2_virt, b"VENTOY_CPIO_HDR");

        let sec3_remapped = substrate.read_bytes_checked(3 * 2048, 8);
        assert_eq!(&sec3_remapped, b"ORIGTAIL");

        let dm_table = substrate.generate_dmsetup_linear_table("/dev/sda");
        assert_eq!(dm_table, "0 8 linear /dev/sda 0");
    }

    impl VirtualSubstrate {
        fn read_bytes_checked(&self, offset: u64, len: usize) -> Vec<u8> {
            self.read_bytes(offset, len).unwrap()
        }
    }
}
